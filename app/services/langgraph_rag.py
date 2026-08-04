"""
Day 6: the same pipeline as langchain_rag.py, restructured as an explicit
LangGraph state graph — query -> retrieve -> generate -> respond.

Why bother, when the LCEL chain already worked? Two reasons, and they're
the actual reason LangGraph exists:

1. Explicit, inspectable state. Each node reads/writes a shared `State`
   dict. You can log, pause, or branch on that state at any point — a
   linear `|` chain can't easily do that.
2. This shape is what Week 2 needs. Once you add a router node (decide
   which agent handles a query) or a second agent (e.g. a "win-likelihood
   scorer" running after retrieval), you're adding nodes and edges to
   *this* graph, not rewriting a chain from scratch. A 3-node graph today
   becomes a multi-agent graph next week without changing the underlying
   pattern.

For a single fixed pipeline like this one, a graph is genuinely more
machinery than necessary — worth being honest about that trade-off if
asked about it.
"""

from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from app.models.gig import Gig
from app.services.langchain_rag import _PROMPT, get_or_build_store
from app.services.llm import generate as llm_generate


class GigMatchState(TypedDict):
    """The shared state every node reads from and writes to.
    Only the DB session is here out of convenience for this project size —
    in a larger app you'd inject dependencies differently rather than
    carry a live session through graph state."""
    profile_text: str
    top_k: int
    db: Session
    docs: list[Document]
    gigs: list[Gig]
    explanation: str


def retrieve_node(state: GigMatchState) -> dict:
    """Node 1: embed the query and retrieve the top-k matching gigs."""
    store = get_or_build_store(state["db"])
    if store is None:
        return {"docs": []}
    retriever = store.as_retriever(search_kwargs={"k": state["top_k"]})
    docs = retriever.invoke(state["profile_text"])
    return {"docs": docs}


def generate_node(state: GigMatchState) -> dict:
    """Node 2: build a grounded prompt from the retrieved docs and call the LLM."""
    docs = state["docs"]
    if not docs:
        return {"explanation": "No matching gigs found in the index yet."}

    context = "\n\n".join(f"[Gig #{d.metadata['gig_id']}] {d.page_content}" for d in docs)
    prompt = _PROMPT.format(profile=state["profile_text"], context=context)
    explanation = llm_generate(prompt)
    return {"explanation": explanation}


def respond_node(state: GigMatchState) -> dict:
    """Node 3: turn retrieved doc metadata back into full Gig rows from
    Postgres, preserving retrieval order, for the final API response."""
    docs = state["docs"]
    if not docs:
        return {"gigs": []}

    gig_ids = [d.metadata["gig_id"] for d in docs]
    gigs = state["db"].query(Gig).filter(Gig.id.in_(gig_ids)).all()
    gigs_by_id = {g.id: g for g in gigs}
    ordered_gigs = [gigs_by_id[gid] for gid in gig_ids if gid in gigs_by_id]
    return {"gigs": ordered_gigs}


def _build_graph():
    builder = StateGraph(GigMatchState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("respond", respond_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "respond")
    builder.add_edge("respond", END)

    return builder.compile()


# Compiled once at import time — cheap to reuse across requests.
gig_match_graph = _build_graph()


def recommend_gigs_langgraph(db: Session, profile_text: str, top_k: int = 5) -> dict:
    """Entry point the API router calls — runs the full graph and returns
    the same shape as the other two pipelines, for direct comparison."""
    result = gig_match_graph.invoke({
        "profile_text": profile_text,
        "top_k": top_k,
        "db": db,
        "docs": [],
        "gigs": [],
        "explanation": "",
    })
    return {"gigs": result["gigs"], "explanation": result["explanation"]}
