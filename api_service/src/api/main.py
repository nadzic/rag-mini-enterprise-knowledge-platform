from fastapi import APIRouter, FastAPI

from api.routes import health, rag

app = FastAPI()
api_router = APIRouter(prefix="/api")

api_router.include_router(health.router)
api_router.include_router(rag.router)
app.include_router(api_router)

@app.get("/api")
async def root():
    return {"message": "Hello World"}