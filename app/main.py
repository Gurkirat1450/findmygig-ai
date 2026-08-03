from fastapi import FastAPI

from app.routers import gigs, health

app = FastAPI(
    title="FindMyGig AI",
    description="AI-powered gig/freelance project-matching platform.",
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(gigs.router, prefix="/gigs", tags=["gigs"])


@app.get("/")
def root():
    return {
        "message": "FindMyGig AI is running.",
        "docs": "/docs",
    }
