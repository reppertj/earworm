from typing import TYPE_CHECKING

from sqlalchemy import Column, ForeignKey, Integer, String, UnicodeText
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
    license_id = Column(Integer, ForeignKey("license.id"))
    license = relationship("License", back_populates="tracks")
    provider_id = Column(Integer, ForeignKey("provider.id"))
    provider = relationship("Provider", back_populates="tracks")
    embeddings = relationship("Embedding", back_populates="track")
