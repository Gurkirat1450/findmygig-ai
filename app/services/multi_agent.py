"""
Day 3, Week 2: a real multi-agent LangGraph — two specialized agents
sharing state, instead of the single-path pipeline from Week 1.

Agent 1: Matcher — the same LLM-based fit explanation from Day 6.
Agent 2: Win-Likelihood Scorer — NEW. Estimates how likely the user is to
          win each retrieved gig, based on:
            - skill overlap ratio (how many of the gig's required skills
              appear in the user's profile text)
            - client activity level (how many gigs this client has posted
              before — a proxy for "established client with a track
              record" vs "one-off poster")
          This agent does NOT call an LLM — it's pure computation over
          data already in Postgres. Worth understanding: "agent" in
          LangGraph just means "a node with a job." Not every node needs
          to reason with an LLM; some just need to DO something with data.
          This is also why this design is quota-friendly: only Matcher
          costs a Gemini call, Scorer is free and instant.

Graph shape (fan-out / fan-in):

    START -> retrieve -> Matcher agent   -\
                       -> Scorer agent    -> respond -> END

Both agents read from the same state (populated by retrieve) and run
independently — LangGraph runs nodes at the same "level" concurrently and
waits for both to finish before respond executes. This is the actual
payoff of using a graph instead of a linear chain: branching + joining
isn't something LCEL's `|` syntax expresses naturally.
"""

from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from app.models.gig import Gig
from app.services.langchain_rag import _PROMPT, get_or_build_store
from app.services.llm import generate as llm_generate


class MultiAgentState(TypedDict):
    profile_text: str
    top_k: int
    db: Session
    docs: list[Document]
    gigs: list[Gig]
    fit_explanation: str
    win_likelihood: list[dict]  # [{gig_id, score, note}, ...]


def retrieve_node(state: MultiAgentState) -> dict:
    """Shared first step: retrieve the top-k matching gigs. Both agents
    downstream read from this same retrieved set."""
    store = get_or_build_store(state["db"])
    if store is None:
        return {"docs": [], "gigs": []}

    retriever = store.as_retriever(search_kwargs={"k": state["top_k"]})
    docs = retriever.invoke(state["profile_text"])

    gig_ids = [d.metadata["gig_id"] for d in docs]
    gigs = state["db"].query(Gig).filter(Gig.id.in_(gig_ids)).all()
    gigs_by_id = {g.id: g for g in gigs}
    ordered_gigs = [gigs_by_id[gid] for gid in gig_ids if gid in gigs_by_id]

    return {"docs": docs, "gigs": ordered_gigs}


def matcher_agent_node(state: MultiAgentState) -> dict:
    """Agent 1: explains WHY each gig is or isn't a fit, grounded in the
    retrieved gigs. Same logic as Day 6's generate_node, now living
    alongside a second agent instead of being the only step."""
    docs = state["docs"]
    if not docs:
        return {"fit_explanation": "No matching gigs found."}

    context = "\n\n".join(f"[Gig #{d.metadata['gig_id']}] {d.page_content}" for d in docs)
    prompt = _PROMPT.format(profile=state["profile_text"], context=context)
    explanation = llm_generate(prompt)
    return {"fit_explanation": explanation}


def win_likelihood_agent_node(state: MultiAgentState) -> dict:
    """Agent 2: scores each gig's win-likelihood — NO LLM call.

    score = 0.7 * skill_overlap_ratio + 0.3 * client_familiarity_bonus

    - skill_overlap_ratio: fraction of the gig's required skills that
      appear (case-insensitive substring match) in the profile text.
    - client_familiarity_bonus: clients with an established posting
      history (2+ gigs) get a small bonus — the reasoning being an
      active client is more likely to actually hire, vs a one-off
      listing that may never close. This is a simplifying assumption,
      not a validated model — worth saying so out loud if asked.
    """
    profile_lower = state["profile_text"].lower()
    results = []

    for gig in state["gigs"]:
        skills = gig.required_skills or []
        if skills:
            matched = sum(1 for s in skills if s.lower() in profile_lower)
            skill_overlap = matched / len(skills)
        else:
            skill_overlap = 0.0

        client_gig_count = (
            state["db"].query(Gig).filter(Gig.client_name == gig.client_name).count()
        )
        client_bonus = min(client_gig_count / 5, 1.0)  # caps out at 5+ prior gigs

        score = round(0.7 * skill_overlap + 0.3 * client_bonus, 3)

        note = (
            f"{matched if skills else 0}/{len(skills)} required skills matched. "
            f"Client '{gig.client_name}' has posted {client_gig_count} gig(s)."
        )
        results.append({"gig_id": gig.id, "score": score, "note": note})

    return {"win_likelihood": results}


def respond_node(state: MultiAgentState) -> dict:
    """Join point: both agents have finished, nothing more to compute —
    this node exists mainly for clarity/extensibility (e.g. a future
    step that re-ranks gigs using both agents' outputs together)."""
    return {}


def _build_graph():
    builder = StateGraph(MultiAgentState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("matcher_agent", matcher_agent_node)
    builder.add_node("win_likelihood_agent", win_likelihood_agent_node)
    builder.add_node("respond", respond_node)

    builder.add_edge(START, "retrieve")
    # Fan-out: both agents run off the same retrieved state.
    builder.add_edge("retrieve", "matcher_agent")
    builder.add_edge("retrieve", "win_likelihood_agent")
    # Fan-in: respond waits for both agents to finish.
    builder.add_edge("matcher_agent", "respond")
    builder.add_edge("win_likelihood_agent", "respond")
    builder.add_edge("respond", END)

    return builder.compile()


multi_agent_graph = _build_graph()


def run_multi_agent(db: Session, profile_text: str, top_k: int = 5) -> dict:
    result = multi_agent_graph.invoke({
        "profile_text": profile_text,
        "top_k": top_k,
        "db": db,
        "docs": [],
        "gigs": [],
        "fit_explanation": "",
        "win_likelihood": [],
    })
    return {
        "gigs": result["gigs"],
        "fit_explanation": result["fit_explanation"],
        "win_likelihood": result["win_likelihood"],
    }
