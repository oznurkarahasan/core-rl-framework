# src/core_rl/api/platform_router.py
#
# FastAPI router for the Visual Learning Platform (Phase 7).
# Mount this onto the existing app with: app.include_router(platform_router)
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

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core_rl.data.db import PlatformDB
from core_rl.data.uploader import ImageUploader

# ------------------------------------------------------------------
# Defaults — can be overridden via create_platform_router()
# ------------------------------------------------------------------

DEFAULT_DB_PATH = Path("platform.db")
DEFAULT_UPLOAD_DIR = Path("uploads")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")


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
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    target_size: tuple[int, int] = (416, 416),
) -> APIRouter:
    """
    Build and return the platform APIRouter.

    Mount onto an existing FastAPI app:
        app.include_router(create_platform_router())

    Args:
        db_path:        Path to the SQLite database file.
        upload_dir:     Directory where uploaded images are stored.
        checkpoint_dir: Directory for model checkpoints and ONNX exports.
        target_size:    Resize target (width, height) for uploaded images.
    """
    db = PlatformDB(path=db_path)
    db.init()

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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

        return UploadResponse(uploaded=len(results), images=results)

    # ----------------------------------------------------------------
    # Serve uploaded images
    # ----------------------------------------------------------------

    @router.get("/image/{filename}", include_in_schema=False)
    def serve_image(filename: str) -> FileResponse:
        """Serve a stored image by filename."""
        image_path = upload_dir / filename
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
        db.add_label(body.image_id, body.category_id, confirmed=body.confirmed)
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

    # ----------------------------------------------------------------
    # Training (Task 7.7)
    # ----------------------------------------------------------------

    # In-memory store for active training jobs: session_id -> job dict
    _training_jobs: dict[int, dict] = {}

    class TrainRequest(BaseModel):
        session_id: int
        epochs: int = 5
        batch_size: int = 16
        lr: float = 1e-4

    class TrainStatusResponse(BaseModel):
        session_id: int
        status: str          # "idle" | "running" | "done" | "error"
        epoch: int | None = None
        total_epochs: int | None = None
        loss: float | None = None
        accuracy: float | None = None
        error: str | None = None

    @router.post("/train", response_model=TrainStatusResponse, status_code=202)
    def start_training(body: TrainRequest) -> TrainStatusResponse:
        """
        Start model training in a background thread.
        Returns immediately with status 'running'.
        Poll GET /platform/train/status?session_id=X for progress.
        """
        import threading
        from core_rl.training.trainer import PlatformTrainer

        session = db.get_session(body.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {body.session_id} not found.")

        if _training_jobs.get(body.session_id, {}).get("status") == "running":
            raise HTTPException(status_code=409, detail="Training already running for this session.")

        _training_jobs[body.session_id] = {
            "status": "running",
            "epoch": 0,
            "total_epochs": body.epochs,
            "loss": None,
            "accuracy": None,
            "error": None,
        }

        def _run() -> None:
            try:
                trainer = PlatformTrainer(
                    db=db,
                    session_id=body.session_id,
                    checkpoint_dir=checkpoint_dir,
                )

                def on_progress(metrics: dict) -> None:
                    _training_jobs[body.session_id].update({
                        "epoch": metrics["epoch"],
                        "loss": metrics["loss"],
                        "accuracy": metrics["accuracy"],
                    })

                trainer.train(
                    epochs=body.epochs,
                    batch_size=body.batch_size,
                    lr=body.lr,
                    on_progress=on_progress,
                )
                _training_jobs[body.session_id]["status"] = "done"
            except Exception as e:
                _training_jobs[body.session_id].update({
                    "status": "error",
                    "error": str(e),
                })

        threading.Thread(target=_run, daemon=True).start()

        return TrainStatusResponse(
            session_id=body.session_id,
            status="running",
            total_epochs=body.epochs,
        )

    @router.get("/train/status", response_model=TrainStatusResponse)
    def train_status(session_id: int) -> TrainStatusResponse:
        """Poll training progress for a session."""
        job = _training_jobs.get(session_id)
        if job is None:
            return TrainStatusResponse(session_id=session_id, status="idle")
        return TrainStatusResponse(session_id=session_id, **job)

    # ----------------------------------------------------------------
    # Export (Task 7.8)
    # ----------------------------------------------------------------

    class ExportResponse(BaseModel):
        path: str
        filename: str
        categories: list[str]
        version: int

    @router.post("/export", response_model=ExportResponse)
    def export_model(session_id: int) -> ExportResponse:
        """
        Export the current checkpoint to ONNX.
        Returns the download path.
        """
        from core_rl.training.trainer import PlatformTrainer

        session = db.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

        if _training_jobs.get(session_id, {}).get("status") == "running":
            raise HTTPException(status_code=409, detail="Training in progress. Wait for it to finish.")

        try:
            trainer = PlatformTrainer(db=db, session_id=session_id, checkpoint_dir=checkpoint_dir)
            # Load existing checkpoint without re-training
            rows = db.get_labeled_data(session_id)
            if not rows:
                raise HTTPException(status_code=422, detail="No labeled data found.")
            trainer.categories = sorted({row["category"] for row in rows})
            trainer._load_or_build_model(len(trainer.categories))
            onnx_path = trainer.export_onnx()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        _predictor_cache.pop(session_id, None)

        return ExportResponse(
            path=f"/platform/download/{onnx_path.name}",
            filename=onnx_path.name,
            categories=trainer.categories,
            version=trainer.version,
        )

    @router.get("/download/{filename}", include_in_schema=False)
    def download_model(filename: str):
        """Download an exported ONNX model."""
        from fastapi.responses import FileResponse as FR
        model_path = checkpoint_dir / filename
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model file not found.")
        return FR(
            str(model_path),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    # ----------------------------------------------------------------
    # Test & Validation (Task 7.9 / 7.10)
    # ----------------------------------------------------------------

    class PredictResponse(BaseModel):
        image_id: int
        filename: str
        path: str           # relative URL: /platform/image/{filename}
        category: str
        confidence: float
        scores: dict[str, float]

    class ValidationSubmit(BaseModel):
        image_id: int
        predicted_category: str
        correct: bool

    class ValidationStatsResponse(BaseModel):
        total: int
        correct: int
        accuracy: float
        per_category: dict[str, dict]       # {cat: {total, correct, accuracy}}
        confusion_matrix: dict[str, dict]   # {actual_cat: {predicted_cat: count}}

    # Predictor cache: session_id -> Predictor (invalidated on new export)
    _predictor_cache: dict[int, object] = {}

    def _get_predictor(session_id: int):
        from core_rl.training.predictor import Predictor
        if session_id not in _predictor_cache:
            _predictor_cache[session_id] = Predictor.from_session(session_id, checkpoint_dir)
        return _predictor_cache[session_id]

    @router.get("/test/next", response_model=PredictResponse | None)
    def test_next(session_id: int) -> PredictResponse | None:
        """Return next unvalidated image with model prediction."""
        session = db.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

        try:
            predictor = _get_predictor(session_id)
        except FileNotFoundError as e:
            raise HTTPException(status_code=422, detail=str(e))

        validated_ids = db.get_validated_image_ids(session_id)

        for img in db.list_images(session_id):
            if img["id"] in validated_ids:
                continue
            try:
                result = predictor.predict(img["path"])
                filename = Path(img["path"]).name
                return PredictResponse(
                    image_id=img["id"],
                    filename=filename,
                    path=f"/platform/image/{filename}",
                    category=result["category"],
                    confidence=result["confidence"],
                    scores=result["scores"],
                )
            except Exception as exc:
                logger.warning("Prediction failed for image %s: %s", img["path"], exc)
                continue

        return None

    @router.post("/test/predict")
    async def predict_upload(
        session_id: int,
        file: UploadFile = File(...),
    ) -> dict:
        """
        Run a one-shot prediction on an uploaded image.
        Does not save the image or record validation results.
        """
        session = db.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

        try:
            predictor = _get_predictor(session_id)
        except FileNotFoundError as e:
            raise HTTPException(status_code=422, detail=str(e))

        try:
            data = await file.read()
            return predictor.predict_bytes(data)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}")

    class AdhocValidationSubmit(BaseModel):
        predicted_category: str
        actual_category: str
        correct: bool

    @router.post("/test/validate-adhoc")
    def submit_adhoc_validation(body: AdhocValidationSubmit, session_id: int) -> dict:
        """Record a correct/wrong judgment on an ad-hoc (uploaded) image prediction."""
        session = db.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
        db.add_adhoc_validation(
            session_id=session_id,
            predicted_cat=body.predicted_category,
            actual_cat=body.actual_category,
            correct=body.correct,
        )
        return {"recorded": True}

    @router.post("/test/reset")
    def reset_validation(session_id: int) -> dict:
        """Clear all validation results for a session (start over)."""
        db.clear_validation_results(session_id)
        db.clear_adhoc_validations(session_id)
        return {"reset": True}

    @router.post("/test/validate")
    def submit_validation(body: ValidationSubmit, session_id: int) -> dict:
        """Submit human judgment on a model prediction."""
        actual_cat = db.get_image_actual_category(body.image_id)
        db.add_validation_result(
            session_id=session_id,
            image_id=body.image_id,
            predicted_cat=body.predicted_category,
            actual_cat=actual_cat,
            correct=body.correct,
        )
        return {"recorded": True}

    @router.get("/test/stats", response_model=ValidationStatsResponse)
    def validation_stats(session_id: int) -> ValidationStatsResponse:
        """Return live accuracy + confusion matrix (session images + ad-hoc uploads)."""
        # Normalize both sources to the same shape: {predicted_cat, actual_cat, correct}
        rows = [
            {"predicted_cat": r["predicted_cat"], "actual_cat": r["actual_cat"], "correct": r["correct"]}
            for r in db.get_validation_results(session_id)
        ] + [
            {"predicted_cat": r["predicted_cat"], "actual_cat": r["actual_cat"], "correct": r["correct"]}
            for r in db.get_adhoc_validations(session_id)
        ]

        total   = len(rows)
        correct = sum(1 for r in rows if r["correct"])
        accuracy = round(correct / total * 100, 1) if total > 0 else 0.0

        per_category: dict[str, dict] = {}
        confusion_matrix: dict[str, dict] = {}

        for r in rows:
            pred   = r["predicted_cat"]
            actual = r["actual_cat"]

            if pred not in per_category:
                per_category[pred] = {"total": 0, "correct": 0, "accuracy": 0.0}
            per_category[pred]["total"] += 1
            if r["correct"]:
                per_category[pred]["correct"] += 1

            if actual:
                if actual not in confusion_matrix:
                    confusion_matrix[actual] = {}
                confusion_matrix[actual][pred] = confusion_matrix[actual].get(pred, 0) + 1

        for cat, s in per_category.items():
            s["accuracy"] = round(s["correct"] / s["total"] * 100, 1) if s["total"] > 0 else 0.0

        return ValidationStatsResponse(
            total=total,
            correct=correct,
            accuracy=accuracy,
            per_category=per_category,
            confusion_matrix=confusion_matrix,
        )

    return router