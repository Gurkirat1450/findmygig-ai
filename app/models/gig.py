from sqlalchemy import ARRAY, Column, Integer, String

from app.database import Base


class Gig(Base):
    __tablename__ = "gigs"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(String, nullable=False)
    required_skills = Column(ARRAY(String), default=list)
    client_name = Column(String, nullable=True)
