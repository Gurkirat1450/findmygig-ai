"""
The RAG pipeline: retrieve relevant gigs (via vector search), build a prompt
grounded in those specific gigs, then ask the LLM to explain the fit.

This is the piece that separates "search" from "RAG" — search alone gives
you ranked results; RAG adds a reasoning layer on top that's grounded in
(cites) the retrieved data, rather than the LLM guessing from general
knowledge. That grounding is exactly what reduces hallucination.
"""

from sqlalchemy.orm import Session

from app.models.gig import Gig
from app.services.embeddings import embed_text
from app.services.llm import generate
from app.services.vector_store import vector_store


def _build_prompt(profile_text: str, gigs: list[Gig]) -> str:
    """
    Construct a grounded prompt: give the LLM ONLY the retrieved gigs as
    context, and instruct it to only reason about those — this is what
    keeps the answer grounded instead of the model inventing gigs that
    don't exist in your database.
    """
    context_blocks = []
    for gig in gigs:
        skills = ", ".join(gig.required_skills or [])
        context_blocks.append(
            f"[Gig #{gig.id}] {gig.title}\n"
            f"Client: {gig.client_name or 'Unknown'}\n"
            f"Required skills: {skills}\n"
            f"Description: {gig.description}"
        )
    context = "\n\n".join(context_blocks)

    return f"""You are FindMyGig AI, an assistant that helps freelance developers find gigs worth applying to.

A user's profile/skills:
"{profile_text}"

Below are the top matching gigs retrieved for this user, based on semantic similarity. Only use these gigs — do not invent gigs that aren't listed.

{context}

For each gig, briefly explain (1-2 sentences) why it is or isn't a strong fit for this user's profile, referencing specific skills that match or are missing. Reference each gig by its [Gig #id] tag. Keep the whole response concise."""


def recommend_gigs(db: Session, profile_text: str, top_k: int = 5) -> dict:
    """
    Full RAG pipeline: embed the query -> retrieve top-k gigs -> build a
    grounded prompt -> generate an explanation.

    Returns both the retrieved gigs (so the caller/UI can show them
    directly) and the LLM's explanation (so the caller gets reasoning,
    not just a ranked list).
    """
    query_embedding = embed_text(profile_text)
    matches = vector_store.search(query_embedding, top_k=top_k)

    gigs = []
    for gig_id, score in matches:
        gig = db.query(Gig).filter(Gig.id == gig_id).first()
        if gig:
            gigs.append(gig)

    if not gigs:
        return {"gigs": [], "explanation": "No matching gigs found in the index yet."}

    prompt = _build_prompt(profile_text, gigs)
    explanation = generate(prompt)

    return {"gigs": gigs, "explanation": explanation}
