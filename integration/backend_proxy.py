"""Optional adapter for the teammate's existing FastAPI backend.

Copy into that backend, then app.include_router(create_ai_router()).
Supply the project's authentication dependencies on include_router as needed.
"""
import os
import httpx
from fastapi import APIRouter, HTTPException, Request


def create_ai_router():
    router = APIRouter(prefix="/api/v1/ai", tags=["AI integration"])
    base_url = os.environ.get("AI_SERVICE_URL", "http://127.0.0.1:8001")

    async def forward(method, path, **kwargs):
        try:
            async with httpx.AsyncClient(base_url=base_url, timeout=10) as client:
                response = await client.request(method, path, **kwargs)
            if response.is_error:
                raise HTTPException(response.status_code, response.json())
            return response.json()
        except httpx.TimeoutException as exc:
            raise HTTPException(504, "AI service timed out; retry without recording a learner error") from exc
        except httpx.RequestError as exc:
            raise HTTPException(503, "AI service unavailable") from exc

    @router.get("/capabilities")
    async def capabilities():
        return await forward("GET", "/api/v1/ai/capabilities")

    @router.post("/predict/landmarks")
    async def landmarks(request: Request):
        return await forward("POST", "/api/v1/ai/predict/landmarks", json=await request.json())

    @router.post("/predict/attempt")
    async def attempt(request: Request):
        return await forward("POST", "/api/v1/ai/predict/attempt", json=await request.json())

    return router
