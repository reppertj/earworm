from fastapi import APIRouter

from app.api.api_v1.endpoints import login, users, utils, tracks, providers, licenses, embeddings

api_router = APIRouter()
api_router.include_router(login.router, tags=["login"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(utils.router, prefix="/utils", tags=["utils"])
api_router.include_router(tracks.router, prefix="/tracks", tags=["tracks"])
api_router.include_router(providers.router, prefix="/providers", tags=["providers"])
api_router.include_router(licenses.router, prefix="/licenses", tags=["licenses"])
api_router.include_router(embeddings.router, prefix="/embeddings", tags=["embeddings"])