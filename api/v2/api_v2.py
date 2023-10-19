from fastapi import APIRouter
from app.api.v2.endpoints import game, health_check, auth

router = APIRouter()

router.include_router(game.router, prefix="/game", tags=["game"])
router.include_router(health_check.router,
                      prefix="/health-check", tags=["health_check"])
router.include_router(auth.router, tags=["auth"])
