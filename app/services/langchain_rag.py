"""
Day 5: the same RAG pipeline from rag.py, rebuilt with LangChain's
abstractions instead of raw sentence-transformers + FAISS calls.

Worth comparing side by side with rag.py:
- embeddings.py + vector_store.py (raw)  -->  HuggingFaceEmbeddings + FAISS.from_documents (LangChain)
- Manual prompt string building          -->  ChatPromptTemplate
- Manual "call the model" function       -->  ChatGoogleGenerativeAI + LCEL chain (the `|` pipe syntax)

Same result, less glue code — but you lose a bit of visibility into what's
happening at each step, which is exactly why building the raw version
first (Days 3-4) before this one is the right order to learn it in.
"""

import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy.orm import Session

from app.models.gig import Gig

load_dotenv()

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # same model as the raw pipeline, for a fair comparison

_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
_vector_store: FAISS | None = None  # built lazily from Postgres, see get_or_build_store()

_PROMPT = ChatPromptTemplate.from_template(
    """You are FindMyGig AI, an assistant that helps freelance developers find gigs worth applying to.

A user's profile/skills:
"{profile}"

Below are the top matching gigs retrieved for this user, based on semantic similarity. Only use these gigs — do not invent gigs that aren't listed.

{context}

For each gig, briefly explain (1-2 sentences) why it is or isn't a strong fit for this user's profile, referencing specific skills that match or are missing. Reference each gig by its [Gig #id] tag. Keep the whole response concise."""
)


def _gig_to_document(gig: Gig) -> Document:
    """Turn a Gig row into a LangChain Document — the unit LangChain's
    retrievers and vector stores operate on."""
    skills = ", ".join(gig.required_skills or [])
    content = f"{gig.title}. {gig.description}. Required skills: {skills}."
    return Document(page_content=content, metadata={"gig_id": gig.id})


def get_or_build_store(db: Session) -> FAISS:
    """
    Build the LangChain FAISS store from Postgres on first use, then reuse it.
    For a project this size, rebuilding per-request would also be fine —
    caching just avoids re-embedding everything on every call.
    """
    global _vector_store
    if _vector_store is None:
        gigs = db.query(Gig).all()
        documents = [_gig_to_document(g) for g in gigs]
        if not documents:
            # FAISS.from_documents errors on an empty list — handle gracefully.
            return None
        _vector_store = FAISS.from_documents(documents, _embeddings)
    return _vector_store


def add_gig_to_store(gig: Gig):
    """Add a single new gig to the existing store, so newly created gigs
    are searchable immediately without a full rebuild."""
    global _vector_store
    doc = _gig_to_document(gig)
    if _vector_store is None:
        _vector_store = FAISS.from_documents([doc], _embeddings)
    else:
        _vector_store.add_documents([doc])


def _format_docs(docs: list[Document]) -> str:
    """Turn retrieved documents back into the same [Gig #id]-tagged context
    block format used in rag.py, so both pipelines produce comparable output."""
    blocks = []
    for doc in docs:
        blocks.append(f"[Gig #{doc.metadata['gig_id']}] {doc.page_content}")
    return "\n\n".join(blocks)


def recommend_gigs_langchain(db: Session, profile_text: str, top_k: int = 5) -> dict:
    """
    The LangChain-native version of recommend_gigs() from rag.py.
    Same inputs/outputs, different plumbing underneath.
    """
    store = get_or_build_store(db)
    if store is None:
        return {"gigs": [], "explanation": "No matching gigs found in the index yet."}

    retriever = store.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(profile_text)

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    # LCEL chain: prompt -> LLM -> parse to plain string.
    # The retrieval step happens above (docs), so it can be reused to also
    # fetch the full Gig rows from Postgres for the response.
    chain = _PROMPT | llm | StrOutputParser()
    explanation = chain.invoke({
        "profile": profile_text,
        "context": _format_docs(docs),
    })

    gig_ids = [doc.metadata["gig_id"] for doc in docs]
    gigs = db.query(Gig).filter(Gig.id.in_(gig_ids)).all()
    # Preserve retrieval order (Postgres IN doesn't guarantee it)
    gigs_by_id = {g.id: g for g in gigs}
    ordered_gigs = [gigs_by_id[gid] for gid in gig_ids if gid in gigs_by_id]

    return {"gigs": ordered_gigs, "explanation": explanation}
