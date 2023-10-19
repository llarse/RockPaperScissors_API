from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import timedelta, datetime
import jwt
from app.core.config import settings
from app.api.dependencies import encode_jwt

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


@router.post("/token/")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    username = form_data.username

    # access_token_expires = timedelta(
    #     minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = encode_jwt(username)
    return {"access_token": access_token, "token_type": "bearer"}
