from fastapi import APIRouter, HTTPException

from app.schemas.gig import GigCreate, GigOut

router = APIRouter()

# TEMPORARY in-memory store — replace with Postgres in Week 1, Day 1-2 of the sprint.
_fake_db: dict[int, GigOut] = {}
_next_id = 1


@router.get("/", response_model=list[GigOut])
def list_gigs():
    """List all gig listings currently in the store."""
    return list(_fake_db.values())


@router.post("/", response_model=GigOut, status_code=201)
def create_gig(gig: GigCreate):
    """Add a new gig listing. Will later be populated by a scraper/ingestion pipeline."""
    global _next_id
    new_gig = GigOut(id=_next_id, **gig.model_dump())
    _fake_db[_next_id] = new_gig
    _next_id += 1
    return new_gig


@router.get("/{gig_id}", response_model=GigOut)
def get_gig(gig_id: int):
    gig = _fake_db.get(gig_id)
    if not gig:
        raise HTTPException(status_code=404, detail="Gig not found")
    return gig
