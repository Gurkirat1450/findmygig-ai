from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["health"])
def health_check():
    """Simple liveness check — useful once this is Dockerized/deployed."""
    return {"status": "ok"}
