# FindMyGig AI

An AI-powered gig/freelance project-matching platform. Agents rank open freelance/gig listings by fit to a developer's skills, experience, and *win-likelihood* — based on profile match and client history — using RAG-based semantic matching over listing and profile data.

**Status:** 🚧 In active development (Week 1 of a 4-week build sprint)

## Why this exists

Freelance/gig platforms dump every open listing on every user. Sorting through them to find the ones actually worth applying to — the ones that match your skills *and* that you're realistically likely to win — is manual, repetitive work. FindMyGig AI automates that triage.

## Tech stack

| Layer | Tech |
|---|---|
| Backend API | FastAPI |
| Orchestration | LangChain, LangGraph |
| Retrieval | Vector DB (FAISS / Chroma) |
| Database | PostgreSQL |
| Containerization | Docker, docker-compose |

## Roadmap

- [x] Project scaffold + FastAPI skeleton
- [x] Postgres schema (gigs table, more to follow: users, applications)
- [x] Embedding pipeline for gig listings (sentence-transformers, local)
- [x] Vector DB storage + semantic retrieval (FAISS)
- [x] `POST /gigs/search` — semantic search: profile text → ranked matching gigs
- [x] `POST /gigs/recommend` — full RAG pipeline: retrieval + grounded LLM explanation (Gemini)
- [ ] LangGraph agent: rank listings by fit + win-likelihood
- [ ] Multi-agent layer (router + matcher + explainer agents)
- [ ] Dockerize + deploy

## Local setup

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd findmygig-ai

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # on Windows PowerShell: venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment template and fill in your values
cp .env.example .env

# 5. Run the API
uvicorn app.main:app --reload
```

Then visit `http://127.0.0.1:8000/docs` for the interactive API docs.

## Project structure

```
findmygig-ai/
├── app/
│   ├── main.py              # FastAPI app entrypoint
│   ├── routers/              # API route definitions
│   ├── models/               # DB models
│   ├── schemas/               # Pydantic request/response schemas
│   └── services/               # Business logic (matching, embeddings, agents)
├── docker/
├── requirements.txt
├── .env.example
└── README.md
```
