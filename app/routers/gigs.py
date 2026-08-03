from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.gig import Gig
from app.schemas.gig import GigCreate, GigOut, GigRecommendResponse, GigSearchQuery, GigSearchResult
from app.services.embeddings import embed_text, gig_to_text
from app.services.rag import recommend_gigs
from app.services.vector_store import vector_store

router = APIRouter()


@router.get("/", response_model=list[GigOut])
def list_gigs(db: Session = Depends(get_db)):
    """List all gig listings currently in the database."""
    return db.query(Gig).all()


@router.post("/", response_model=GigOut, status_code=201)
def create_gig(gig: GigCreate, db: Session = Depends(get_db)):
    """Add a new gig listing, and embed it into the vector index for search."""
    new_gig = Gig(**gig.model_dump())
    db.add(new_gig)
    db.commit()
    db.refresh(new_gig)

    # Embed + index — so it's immediately searchable, no separate step needed.
    text = gig_to_text(new_gig.title, new_gig.description, new_gig.required_skills or [])
    embedding = embed_text(text)
    vector_store.add(new_gig.id, embedding)

    return new_gig


@router.get("/{gig_id}", response_model=GigOut)
def get_gig(gig_id: int, db: Session = Depends(get_db)):
    gig = db.query(Gig).filter(Gig.id == gig_id).first()
    if not gig:
        raise HTTPException(status_code=404, detail="Gig not found")
    return gig


@router.post("/search", response_model=list[GigSearchResult])
def search_gigs(query: GigSearchQuery, db: Session = Depends(get_db)):
    """
    Semantic search: given a free-text profile/skills description, return
    the most similar gigs ranked by similarity score.

    This is the core of FindMyGig AI — everything after this (win-likelihood
    ranking, agentic re-ranking) builds on top of this retrieval step.
    """
    query_embedding = embed_text(query.profile_text)
    matches = vector_store.search(query_embedding, top_k=query.top_k)

    results = []
    for gig_id, score in matches:
        gig = db.query(Gig).filter(Gig.id == gig_id).first()
        if gig:
            results.append(GigSearchResult(gig=gig, similarity_score=round(score, 4)))
    return results


@router.post("/recommend", response_model=GigRecommendResponse)
def recommend(query: GigSearchQuery, db: Session = Depends(get_db)):
    """
    Full RAG pipeline: retrieve matching gigs + generate a grounded
    explanation of why each one is (or isn't) a good fit.

    This is the difference between "search" and "RAG" — search alone
    returns ranked results; this adds a reasoning layer grounded in
    those specific results, citing them by [Gig #id] rather than
    inventing anything outside the retrieved context.
    """
    try:
        return recommend_gigs(db, query.profile_text, top_k=query.top_k)
    except Exception as e:
        # Surface the real error in the response instead of a generic 500 —
        # makes debugging LLM/API-key issues much faster during development.
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
