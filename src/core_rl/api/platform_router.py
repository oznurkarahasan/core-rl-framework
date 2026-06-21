#
# FastAPI router for the Visual Learning Platform.
#
# Endpoints:
#   POST /platform/sessions               — create a new training session
#   GET  /platform/sessions               — list all sessions
#   POST /platform/sessions/{id}/upload   — upload images into a session
#   POST /platform/categories             — add a category to a session
#   GET  /platform/categories             — list categories for a session
#   GET  /platform/next                   — get next unlabeled image
#   POST /platform/label                  — submit a label
#   GET  /platform/stats                  — session stats

import sqlite3
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core_rl.data.db import PlatformDB
from core_rl.data.uploader import ImageUploader

# ------------------------------------------------------------------
# Defaults — can be overridden via create_platform_router()
# ------------------------------------------------------------------

DEFAULT_DB_PATH = Path("platform.db")
DEFAULT_UPLOAD_DIR = Path("uploads")


# ------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------

class SessionCreate(BaseModel):
    name: str


class SessionResponse(BaseModel):
    id: int
    name: str
    created_at: str
    checkpoint_path: str | None


class CategoryCreate(BaseModel):
    session_id: int
    name: str


class CategoryResponse(BaseModel):
    id: int
    name: str


class LabelSubmit(BaseModel):
    image_id: int
    category_id: int
    confirmed: bool = True


class LabelResponse(BaseModel):
    image_id: int
    category_id: int
    confirmed: bool


class UploadResponse(BaseModel):
    uploaded: int
    images: list[dict[str, Any]]
    errors: list[dict[str, Any]] = []


class NextImageResponse(BaseModel):
    image_id: int
    filename: str
    path: str          # relative URL: /platform/image/{filename}


class StatsResponse(BaseModel):
    total_images: int
    total_labels: int
    per_category: list[dict[str, Any]]


# ------------------------------------------------------------------
# Router factory
# ------------------------------------------------------------------

def create_platform_router(
    db_path: Path = DEFAULT_DB_PATH,
    upload_dir: Path = DEFAULT_UPLOAD_DIR,
    target_size: tuple[int, int] = (416, 416),
) -> APIRouter:
    """
    Build and return the platform APIRouter.

    Mount onto an existing FastAPI app:
        app.include_router(create_platform_router())

    Args:
        db_path:     Path to the SQLite database file.
        upload_dir:  Directory where uploaded images are stored.
        target_size: Resize target (width, height) for uploaded images.
    """
    db = PlatformDB(path=db_path)
    db.init()

    uploader = ImageUploader(upload_dir=upload_dir, target_size=target_size)

    router = APIRouter(prefix="/platform", tags=["platform"])

    # ----------------------------------------------------------------
    # Sessions
    # ----------------------------------------------------------------

    @router.post("/sessions", response_model=SessionResponse, status_code=201)
    def create_session(body: SessionCreate) -> SessionResponse:
        """Create a new training session."""
        sid = db.create_session(body.name)
        row = db.get_session(sid)
        return SessionResponse(
            id=row["id"],
            name=row["name"],
            created_at=row["created_at"],
            checkpoint_path=row["checkpoint_path"],
        )

    @router.get("/sessions", response_model=list[SessionResponse])
    def list_sessions() -> list[SessionResponse]:
        """List all sessions, newest first."""
        rows = db.list_sessions()
        return [
            SessionResponse(
                id=r["id"],
                name=r["name"],
                created_at=r["created_at"],
                checkpoint_path=r["checkpoint_path"],
            )
            for r in rows
        ]

    # ----------------------------------------------------------------
    # Image upload
    # ----------------------------------------------------------------

    @router.post("/sessions/{session_id}/upload", response_model=UploadResponse)
    async def upload_images(
        session_id: int,
        files: list[UploadFile] = File(...),
    ) -> UploadResponse:
        """
        Upload one or more images into a session.
        Images are resized to target_size (letterbox) and saved as JPEG.
        """
        session = db.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

        results = []
        errors = []

        for file in files:
            try:
                file_bytes = await file.read()
                result = uploader.process(file_bytes, file.filename or "upload.jpg")
                image_id = db.add_image(session_id, result.filename, str(result.path))
                results.append({
                    "image_id": image_id,
                    "filename": result.filename,
                    "original_name": result.original_name,
                    "width": result.width,
                    "height": result.height,
                })
            except ValueError as e:
                errors.append({"file": file.filename, "error": str(e)})

        if errors and not results:
            raise HTTPException(
                status_code=422,
                detail={"message": "All uploads failed.", "errors": errors},
            )

        return UploadResponse(uploaded=len(results), images=results, errors=errors)

    # ----------------------------------------------------------------
    # Serve uploaded images
    # ----------------------------------------------------------------

    @router.get("/image/{filename}", include_in_schema=False)
    def serve_image(filename: str) -> FileResponse:
        """Serve a stored image by filename."""
        image_path = (upload_dir / filename).resolve()
        if not image_path.is_relative_to(upload_dir.resolve()):
            raise HTTPException(status_code=403, detail="Access denied.")
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found.")
        return FileResponse(str(image_path), media_type="image/jpeg")

    # ----------------------------------------------------------------
    # Categories
    # ----------------------------------------------------------------

    @router.post("/categories", response_model=CategoryResponse, status_code=201)
    def add_category(body: CategoryCreate) -> CategoryResponse:
        """Add a category to a session. Duplicate names are safely ignored."""
        session = db.get_session(body.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {body.session_id} not found.")

        name = body.name.strip()
        if not name:
            raise HTTPException(status_code=422, detail="Category name cannot be empty.")

        cid = db.add_category(body.session_id, name)
        return CategoryResponse(id=cid, name=name)

    @router.get("/categories", response_model=list[CategoryResponse])
    def list_categories(session_id: int) -> list[CategoryResponse]:
        """List all categories for a session."""
        session = db.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

        rows = db.list_categories(session_id)
        return [CategoryResponse(id=r["id"], name=r["name"]) for r in rows]

    # ----------------------------------------------------------------
    # Labeling
    # ----------------------------------------------------------------

    @router.get("/next", response_model=NextImageResponse | None)
    def next_image(session_id: int, category_id: int) -> NextImageResponse | None:
        """
        Return the next unlabeled image for a given session + category.
        Returns null when all images have been labeled.
        """
        if db.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

        rows = db.get_unlabeled_images(session_id, category_id, limit=1)
        if not rows:
            return None

        row = rows[0]
        filename = Path(row["path"]).name
        return NextImageResponse(
            image_id=row["id"],
            filename=filename,
            path=f"/platform/image/{filename}",
        )

    @router.post("/label", response_model=LabelResponse)
    def submit_label(body: LabelSubmit) -> LabelResponse:
        """Submit a human label for an image."""
        try:
            db.add_label(body.image_id, body.category_id, confirmed=body.confirmed)
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=422, detail="Invalid image_id or category_id.")
        return LabelResponse(
            image_id=body.image_id,
            category_id=body.category_id,
            confirmed=body.confirmed,
        )

    # ----------------------------------------------------------------
    # Stats
    # ----------------------------------------------------------------

    @router.get("/stats", response_model=StatsResponse)
    def get_stats(session_id: int) -> StatsResponse:
        """Return labeling progress stats for a session."""
        session = db.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

        stats = db.get_stats(session_id)
        return StatsResponse(**stats)

    return router