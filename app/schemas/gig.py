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
