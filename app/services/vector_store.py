"""
A minimal FAISS vector index for gig listings.

Design choice: this rebuilds the index from Postgres on startup, and updates
it in-memory as gigs are added. That's fine for a portfolio project and even
a small real deployment. Once the number of gigs gets large (many thousands+),
you'd move to a persisted/incrementally-updated index (FAISS supports saving
to disk with faiss.write_index) or a managed vector DB (Pinecone, Weaviate,
Qdrant) instead — worth knowing that tradeoff for interviews.
"""

import faiss
import numpy as np

EMBEDDING_DIM = 384  # matches all-MiniLM-L6-v2's output size


class GigVectorStore:
    def __init__(self):
        # IndexFlatIP = exact search via inner product; since our embeddings
        # are normalized, inner product == cosine similarity.
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.gig_ids: list[int] = []  # parallel array: index position -> gig id

    def add(self, gig_id: int, embedding: list[float]):
        vector = np.array([embedding], dtype="float32")
        self.index.add(vector)
        self.gig_ids.append(gig_id)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[tuple[int, float]]:
        """Returns [(gig_id, similarity_score), ...] sorted best-match first."""
        if self.index.ntotal == 0:
            return []
        query_vector = np.array([query_embedding], dtype="float32")
        scores, positions = self.index.search(query_vector, min(top_k, self.index.ntotal))
        results = []
        for score, pos in zip(scores[0], positions[0]):
            if pos == -1:
                continue
            results.append((self.gig_ids[pos], float(score)))
        return results

    def reset(self):
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.gig_ids = []


# Single shared instance for the app's lifetime.
# Fine for a single-process dev server; a production deployment with multiple
# workers would need this to live in a shared/persisted store instead.
vector_store = GigVectorStore()
