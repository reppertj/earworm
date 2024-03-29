from typing import TYPE_CHECKING

from sqlalchemy import Column, ForeignKey, Integer, String, UnicodeText, Boolean
from sqlalchemy.orm import relationship

from app.db.base_class import Base

if TYPE_CHECKING:
    from .license import License  # noqa: F401
    from .provider import Provider  # noqa: F401
    from .embedding import Embedding  # noqa: F401


class Track(Base):
    id = Column(Integer, primary_key=True, index=True)
    title = Column(UnicodeText)
    artist = Column(UnicodeText)
    s3_preview_key = Column(String)
    url = Column(String, nullable=False)
    media_url = Column(String)
    active = Column(Boolean)
    license_id = Column(Integer, ForeignKey("license.id"))
    license = relationship("License", uselist=False, back_populates="tracks")
    provider_id = Column(Integer, ForeignKey("provider.id"))
    provider = relationship("Provider", uselist=False, back_populates="tracks")
    embeddings = relationship("Embedding", back_populates="track")
