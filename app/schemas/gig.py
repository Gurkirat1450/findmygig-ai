from pydantic import BaseModel, Field


class GigBase(BaseModel):
    title: str = Field(..., examples=["Build a React dashboard for analytics"])
    description: str = Field(..., examples=["Looking for a dev to build..."])
    required_skills: list[str] = Field(default_factory=list, examples=[["React", "TypeScript"]])
    client_name: str | None = None


class GigCreate(GigBase):
    pass


class GigOut(GigBase):
    id: int

    class Config:
        from_attributes = True


class GigSearchQuery(BaseModel):
    profile_text: str = Field(
        ...,
        examples=["Python developer with FastAPI, LangChain, and RAG experience"],
        description="Free-text description of the user's skills/experience to match against.",
    )
    top_k: int = Field(default=5, ge=1, le=20)


class GigSearchResult(BaseModel):
    gig: GigOut
    similarity_score: float


class GigRecommendResponse(BaseModel):
    gigs: list[GigOut]
    explanation: str
