from pydantic import BaseModel

class HealthResponse(BaseModel):
  status: str = "ok"
  service: str = "api_service"
  version: str = "0.1.0"