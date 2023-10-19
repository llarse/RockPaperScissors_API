from fastapi import FastAPI
from app.core.config import settings
from app.api.v1.api_v1 import router as api_v1_router
from app.api.v2.api_v2 import router as api_v2_router


app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)

app.include_router(api_v1_router, prefix="/v1")
app.include_router(api_v2_router, prefix="/v2")


@app.get("/")
def read_root():
    return {"name": settings.APP_NAME, "version": settings.APP_VERSION}
