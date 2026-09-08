"""
Day 4, Week 2: the same two-agent flow from Day 3 (multi_agent.py),
reimplemented in CrewAI instead of LangGraph.

The key orchestration difference, and the actual point of this file:

LangGraph (Day 3): a node can be ANY Python function. The Win-Likelihood
Scorer was pure arithmetic — zero LLM calls, instant, free. LangGraph
doesn't care whether a node "reasons" or just computes.

CrewAI: every Agent is built around an LLM role (role, goal, backstory).
Even a task that's really just "run this formula" gets wrapped in an LLM
call to decide to use a tool and to phrase the result. That makes this
CrewAI version slower and more expensive (2-3 Gemini calls per run,
vs. 1 for the LangGraph version) for equivalent output — the cost isn't
CrewAI being worse, it's a direct consequence of its every-agent-is-an-LLM
design philosophy. Worth knowing which tradeoff you're choosing and why,
not just that "LangGraph is cheaper" as a fact to memorize.

Retrieval (the vector search step) happens OUTSIDE CrewAI here, reusing
the same get_or_build_store from Day 5 — CrewAI orchestrates the
reasoning agents, not the retrieval plumbing.
"""

import os

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from sqlalchemy.orm import Session

from app.models.gig import Gig
from app.services.langchain_rag import get_or_build_store

os.environ.setdefault("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
_LLM = "gemini/gemini-3.5-flash"  # via litellm; workaround for an open CrewAI bug
                                   # (github.com/crewAIInc/crewAI/issues/6984) — CrewAI's
                                   # native Gemini message formatting doesn't guard against
                                   # a conversation ending on a "model" turn, which Gemini
                                   # 3.6 strictly rejects (Google removed support for
                                   # prefilled model turns). gemini-3.5-flash's older API
                                   # still tolerates it.


@tool("score_win_likelihood")
def score_win_likelihood_tool(gig_id: int, profile_text: str) -> str:
    """Compute a win-likelihood score for a specific gig ID, given the
    user's profile text. Returns skill overlap and client activity data
    the agent should use to justify its answer, not just state a number."""
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        gig = db.query(Gig).filter(Gig.id == gig_id).first()
        if not gig:
            return f"Gig #{gig_id} not found."

        skills = gig.required_skills or []
        profile_lower = profile_text.lower()
        matched = sum(1 for s in skills if s.lower() in profile_lower)
        skill_overlap = matched / len(skills) if skills else 0.0

        client_gig_count = db.query(Gig).filter(Gig.client_name == gig.client_name).count()
        client_bonus = min(client_gig_count / 5, 1.0)

        score = round(0.7 * skill_overlap + 0.3 * client_bonus, 3)
        return (
            f"Gig #{gig_id} ({gig.title}): score={score}. "
            f"{matched}/{len(skills)} skills matched. "
            f"Client '{gig.client_name}' has {client_gig_count} prior gig(s)."
        )
    finally:
        db.close()


def _build_crew(gig_ids: list[int], profile_text: str) -> Crew:
    matcher = Agent(
        role="Gig Matching Specialist",
        goal="Explain, for each given gig, why it is or isn't a strong fit for the user's profile.",
        backstory=(
            "You help freelance developers understand which gigs are worth their time, "
            "by comparing required skills against the user's stated profile."
        ),
        llm=_LLM,
        verbose=False,
    )

    scorer = Agent(
        role="Win-Likelihood Analyst",
        goal="Score each gig's win-likelihood using the score_win_likelihood tool, and briefly justify each score.",
        backstory=(
            "You assess how competitive a freelancer's application would be for a given gig, "
            "based on skill overlap and how active/established the posting client is."
        ),
        tools=[score_win_likelihood_tool],
        llm=_LLM,
        verbose=False,
    )

    gig_ids_str = ", ".join(str(i) for i in gig_ids)

    matcher_task = Task(
        description=(
            f"The user's profile: \"{profile_text}\"\n"
            f"Gig IDs to evaluate: {gig_ids_str}\n"
            "For each gig, briefly explain why it is or isn't a strong fit. "
            "Keep the whole response concise."
        ),
        expected_output="A short per-gig fit explanation, referencing each gig by ID.",
        agent=matcher,
    )

    scorer_task = Task(
        description=(
            f"The user's profile: \"{profile_text}\"\n"
            f"Gig IDs to evaluate: {gig_ids_str}\n"
            "Use the score_win_likelihood tool once per gig ID to compute each score, "
            "then summarize all scores in one final answer."
        ),
        expected_output="A list of gig IDs with their win-likelihood scores and brief justifications.",
        agent=scorer,
    )

    return Crew(
        agents=[matcher, scorer],
        tasks=[matcher_task, scorer_task],
        process=Process.sequential,
        verbose=False,
    )


def run_crewai_multi_agent(db: Session, profile_text: str, top_k: int = 3) -> dict:
    """
    Entry point mirroring run_multi_agent() from Day 3, for direct
    comparison. Note top_k defaults lower here (3, not 5) — deliberately,
    since each additional gig means another tool call inside the crew,
    and this version already costs more LLM calls per run than Day 3's.
    """
    store = get_or_build_store(db)
    if store is None:
        return {"gigs": [], "matcher_output": "No gigs indexed yet.", "scorer_output": ""}

    retriever = store.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(profile_text)
    gig_ids = [d.metadata["gig_id"] for d in docs]

    gigs = db.query(Gig).filter(Gig.id.in_(gig_ids)).all()
    gigs_by_id = {g.id: g for g in gigs}
    ordered_gigs = [gigs_by_id[gid] for gid in gig_ids if gid in gigs_by_id]

    if not ordered_gigs:
        return {"gigs": [], "matcher_output": "No matching gigs found.", "scorer_output": ""}

    crew = _build_crew(gig_ids, profile_text)
    result = crew.kickoff()

    task_outputs = result.tasks_output
    matcher_output = str(task_outputs[0].raw) if len(task_outputs) > 0 else ""
    scorer_output = str(task_outputs[1].raw) if len(task_outputs) > 1 else str(result)

    return {"gigs": ordered_gigs, "matcher_output": matcher_output, "scorer_output": scorer_output}
