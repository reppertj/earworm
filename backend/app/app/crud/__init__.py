from .crud_user import user
from .crud_embedding import embedding
from .crud_embedding_model import embedding_model
from .crud_license import license
from .crud_provider import provider
from .crud_track import track

# For a new basic set of CRUD operations you could just do

# from .base import CRUDBase
# from app.models.item import Item
# from app.schemas.item import ItemCreate, ItemUpdate

# item = CRUDBase[Item, ItemCreate, ItemUpdate](Item)
