"""
Day 2, Week 2: semantic routing.

Theory recap: rule-based routing means hardcoded if/else on keywords
("if 'docker' in query: use tool"). That's brittle — it breaks on
rephrasing ("what needs containerization" wouldn't match "docker").
Embedding-based routing instead compares the MEANING of the query against
example utterances for each route, so it generalizes to phrasing it's
never seen.

This router has two routes:
- "tool_lookup"  -> exact, structured questions (skill filters, client
                    history) -> handled by the Day 1 tool-calling agent
- "recommend"    -> fuzzy, profile-matching questions -> handled by the
                    LangGraph RAG pipeline from Day 6

Each route is represented by a handful of example utterances. The
incoming query is embedded once, compared against every example via
cosine similarity, and routed to whichever route's best-matching example
scores highest. Simple, but this is genuinely how many production
semantic routers work at small scale — before reaching for a dedicated
classifier becomes worth the extra complexity.
"""

import numpy as np
from sqlalchemy.orm import Session

from app.services.agent_tools import chat_with_tools
from app.services.embeddings import embed_text
from app.services.langgraph_rag import recommend_gigs_langgraph

ROUTE_EXAMPLES = {
    "tool_lookup": [
        "What gigs need Docker?",
        "Show me gigs requiring Python",
        "Has Beta Labs posted gigs before?",
        "What is Acme Corp's posting history?",
        "List all gigs that need SQL",
        "Which gigs want React skills?",
    ],
    "recommend": [
        "Find gigs that match my skills",
        "What gigs are a good fit for a Python developer with RAG experience?",
        "Recommend gigs for someone with FastAPI and backend experience",
        "I'm a machine learning engineer, what should I apply to?",
        "Which of these gigs am I most qualified for?",
        "Suggest projects based on my profile",
    ],
}

# Precompute embeddings for every example once at import time — this is
# the "index" the router searches against, same idea as the gig vector
# index, just for routing examples instead of gig listings.
_ROUTE_EMBEDDINGS = {
    route: [embed_text(example) for example in examples]
    for route, examples in ROUTE_EXAMPLES.items()
}


def _cosine_sim(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def classify_route(query: str) -> tuple[str, float]:
    """Returns (route_name, confidence_score) — the route whose best
    matching example is closest to the query."""
    query_embedding = embed_text(query)

    best_route, best_score = None, -1.0
    for route, example_embeddings in _ROUTE_EMBEDDINGS.items():
        score = max(_cosine_sim(query_embedding, ex) for ex in example_embeddings)
        if score > best_score:
            best_route, best_score = route, score

    return best_route, best_score


def route_query(db: Session, query: str, top_k: int = 5) -> dict:
    """
    Classify the query, dispatch to the matching pipeline, and return a
    unified response that also reports which route was chosen — useful
    for debugging/demoing, since it makes the routing decision visible
    instead of a black box.
    """
    route, confidence = classify_route(query)

    if route == "tool_lookup":
        answer = chat_with_tools(query)
        return {"route": route, "confidence": round(confidence, 4), "response": answer}

    # route == "recommend"
    result = recommend_gigs_langgraph(db, query, top_k=top_k)
    return {
        "route": route,
        "confidence": round(confidence, 4),
        "response": result["explanation"],
        "gigs": result["gigs"],
    }
