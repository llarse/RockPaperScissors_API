import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "Rock Paper Scissors API"
    APP_VERSION: str = "2.0.0"

    # FastAPI settings
    DEBUG: bool = True  # Set to False in production
    RELOAD: bool = True  # Set to False in production

    # JWT settings
    SECRET_KEY: str = "LynAPIsecretKeyButNotSoSecret"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"


settings = Settings()
