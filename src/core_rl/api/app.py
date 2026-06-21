import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core_rl.active_learning.review_queue import ReviewQueue

_STATIC = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ResolveRequest(BaseModel):
    reward: float


class QueueStatusResponse(BaseModel):
    pending: int
    capacity: int
    fill_rate: float


class ResolveResponse(BaseModel):
    id: str
    reward: float
    status: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize(value: Any) -> Any:
    """Recursively convert numpy arrays to nested lists for JSON serialisation."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    return value


def _serialize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {**item, "state": _serialize(item.get("state"))}


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    queue: ReviewQueue,
    on_resolve: Optional[Callable[[str, float], None]] = None,
) -> FastAPI:
    """
    Create the FastAPI application.

    Args:
        queue: The ReviewQueue instance shared with the simulation loop.
        on_resolve: Optional callback invoked after a successful resolve.
                    Signature: ``on_resolve(item_id: str, new_reward: float) -> None``.
                    Wire the reward-update bridge here:
                    ``on_resolve=lambda id, r: buffer.update_reward(id, r)``

    Auth:
        Set ``AL_API_KEY`` env variable to enable API key authentication.
        When set, every queue endpoint requires an ``X-API-Key`` header.
        Leave unset (or empty) for local development — no auth required.

    Example::

        app = create_app(queue, on_resolve=lambda id, r: buffer.update_reward(id, r))
        uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    app = FastAPI(
        title="Core-RL Active Learning API",
        description="Human-in-the-loop review queue for uncertain agent states.",
        version="0.1.0",
    )

    api_key = os.getenv("AL_API_KEY", "").strip()

    def _verify_key(x_api_key: str = Header(default="")) -> None:
        if api_key and x_api_key != api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    auth = [Depends(_verify_key)] if api_key else []

    # ------------------------------------------------------------------
    # / — serve the human review UI (no auth)
    # ------------------------------------------------------------------

    from core_rl.api.platform_router import create_platform_router
    app.include_router(create_platform_router())

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_STATIC / "index.html")

    @app.get("/annotate", include_in_schema=False)
    def annotate() -> FileResponse:
        return FileResponse(_STATIC / "annotate.html")

    # ------------------------------------------------------------------
    # /health — no auth (liveness probe, monitoring tools)
    # ------------------------------------------------------------------

    @app.get("/health", tags=["meta"])
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # Queue endpoints
    # ------------------------------------------------------------------

    router = APIRouter(prefix="/queue", tags=["queue"], dependencies=auth)

    @router.get("/status", response_model=QueueStatusResponse)
    def queue_status() -> QueueStatusResponse:
        pending = len(queue)
        return QueueStatusResponse(
            pending=pending,
            capacity=queue.capacity,
            fill_rate=round(pending / queue.capacity, 4),
        )

    @router.get("/items", response_model=List[Dict[str, Any]])
    def queue_items() -> List[Dict[str, Any]]:
        return [_serialize_item(item) for item in queue.get()]

    @router.post("/resolve/{item_id}", response_model=ResolveResponse)
    def resolve_item(item_id: str, body: ResolveRequest) -> ResolveResponse:
        resolved = queue.resolve(item_id, reward=body.reward)
        if resolved is None:
            raise HTTPException(
                status_code=404,
                detail=f"Item '{item_id}' not found. It may have been already resolved or evicted.",
            )
        if on_resolve is not None:
            on_resolve(item_id, body.reward)
        return ResolveResponse(id=item_id, reward=body.reward, status="resolved")

    app.include_router(router)
    return app