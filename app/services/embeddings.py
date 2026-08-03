"""
Embedding generation for gig listings and user profiles.

Uses sentence-transformers (local, free, no API key) so you're not blocked
on API keys this early. Swap out for OpenAI/Gemini embeddings later if you
want higher-quality vectors — the rest of the pipeline (FAISS index, search)
doesn't need to change, since it just works with whatever vector this returns.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for semantic similarity


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the model once and cache it — loading it per-request would be slow."""
    return SentenceTransformer(MODEL_NAME)


def embed_text(text: str):
    """Turn a single string into an embedding vector (list[float])."""
    model = get_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def gig_to_text(title: str, description: str, required_skills: list[str]) -> str:
    """
    Turn a gig's structured fields into one text blob for embedding.
    Keeping this as its own function means the "how we represent a gig
    for embedding" logic lives in one place — you'll likely tweak this
    (e.g. weight skills more heavily) as you iterate.
    """
    skills_text = ", ".join(required_skills)
    return f"{title}. {description}. Required skills: {skills_text}."
