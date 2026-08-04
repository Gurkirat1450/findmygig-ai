from fastapi import FastAPI

from app.database import Base, SessionLocal, engine
from app.models import gig as gig_model  # noqa: F401 — import so SQLAlchemy registers the model
from app.routers import gigs, health
from app.services import langchain_rag
from app.services.embeddings import embed_text, gig_to_text
from app.services.vector_store import vector_store

app = FastAPI(
    title="FindMyGig AI",
    description="AI-powered gig/freelance project-matching platform.",
    version="0.1.0",
)

# Creates tables if they don't exist yet — fine for early dev,
# swap for Alembic migrations once the schema stabilizes.
Base.metadata.create_all(bind=engine)


@app.on_event("startup")
def rebuild_vector_index():
    """
    Rebuild the FAISS index from whatever's already in Postgres on startup.
    Without this, gigs inserted via SQL (like your sql_practice.sql sample
    data) wouldn't be searchable until re-created through the API.
    """
    db = SessionLocal()
    try:
        vector_store.reset()
        for g in db.query(gig_model.Gig).all():
            text = gig_to_text(g.title, g.description, g.required_skills or [])
            vector_store.add(g.id, embed_text(text))
        # Also build the Day 5 LangChain-based store from the same data.
        langchain_rag.get_or_build_store(db)
    finally:
        db.close()

app.include_router(health.router)
app.include_router(gigs.router, prefix="/gigs", tags=["gigs"])


@app.get("/")
def root():
    return {
        "message": "FindMyGig AI is running.",
        "docs": "/docs",
    }
