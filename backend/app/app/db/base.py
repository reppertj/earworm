# Import all the models, so that Base has them before being
# imported by Alembic
from app.db.base_class import Base  # noqa
from app.models.user import User  # noqa
from app.models.embedding import Embedding  # noqa
from app.models.embedding_model import Embedding_Model  # noqa
from app.models.track import Track  # noqa
from app.models.license import License  # noqa
from app.models.provider import Provider  # noqa
