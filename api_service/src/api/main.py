import sys
from pathlib import Path

from fastapi import APIRouter, FastAPI
from api.routes import health, rag

# Allow api_service to import shared project modules (e.g. `services`).
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

app = FastAPI()
api_router = APIRouter(prefix="/api")

api_router.include_router(health.router)
api_router.include_router(rag.router)
app.include_router(api_router)

@app.get("/api")
async def root():
    return {"message": "Hello World"}