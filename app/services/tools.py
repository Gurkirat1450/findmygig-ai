"""
Day 1, Week 2: give the LLM callable tools instead of just retrieval.

The difference from RAG: retrieval (search/recommend) always returns
"semantically similar" results — good for fuzzy matching, bad for precise
questions like "show me EXACTLY the gigs needing Docker" or "has this
client posted before." Tools let the model make an exact, structured
lookup when that's what the question actually needs, and choose RAG
retrieval when it doesn't. That choice — which one to use — is exactly
what Day 2's semantic routing formalizes.

Each tool opens its own short-lived DB session rather than depending on
a request-scoped session, since tools get called by the LLM's tool-calling
loop, not directly by a FastAPI route with a `Depends(get_db)` session
already in hand.
"""

from langchain_core.tools import tool

from app.database import SessionLocal
from app.models.gig import Gig


@tool
def filter_gigs_by_skill(skill: str) -> str:
    """Return all gigs that explicitly require a given skill (exact match,
    not semantic similarity). Use this when the user asks for gigs
    requiring a SPECIFIC named technology, e.g. 'show me Docker gigs'."""
    db = SessionLocal()
    try:
        gigs = db.query(Gig).filter(Gig.required_skills.any(skill)).all()
        if not gigs:
            return f"No gigs found requiring '{skill}'."
        lines = [f"[Gig #{g.id}] {g.title} — client: {g.client_name}" for g in gigs]
        return "\n".join(lines)
    finally:
        db.close()


@tool
def get_client_gig_history(client_name: str) -> str:
    """Return every gig a specific client has posted, so the agent can
    reason about that client's posting patterns (e.g. what kind of work
    they typically hire for). This is the foundation for a future
    win-likelihood score based on client history."""
    db = SessionLocal()
    try:
        gigs = db.query(Gig).filter(Gig.client_name == client_name).all()
        if not gigs:
            return f"No gig history found for client '{client_name}'."
        lines = [f"[Gig #{g.id}] {g.title} — skills: {', '.join(g.required_skills or [])}" for g in gigs]
        return f"{client_name} has posted {len(gigs)} gig(s):\n" + "\n".join(lines)
    finally:
        db.close()


ALL_TOOLS = [filter_gigs_by_skill, get_client_gig_history]
