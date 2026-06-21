# tests/test_platform_train.py
#
# Test suite for /platform/train and /platform/export endpoints (Task 7.7 & 7.8)

import io
import time
import json
import pytest
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import models
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core_rl.api.platform_router import create_platform_router
from core_rl.training.trainer import PlatformTrainer


# ------------------------------------------------------------------
# Patch: skip pretrained weight download
# ------------------------------------------------------------------

def _build_model_no_pretrain(self, num_classes):
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model.to(self.device)


@pytest.fixture(autouse=True)
def no_pretrain(monkeypatch):
    monkeypatch.setattr(PlatformTrainer, "_build_model", _build_model_no_pretrain)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def make_jpeg(path: Path, color=(100, 150, 200)) -> None:
    Image.new("RGB", (64, 64), color=color).save(path, format="JPEG")


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    app = FastAPI()
    router = create_platform_router(
        db_path=tmp_path / "test.db",
        upload_dir=tmp_path / "uploads",
        checkpoint_dir=tmp_path / "checkpoints",
        target_size=(64, 64),
    )
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def ready_session(client: TestClient, tmp_path: Path) -> dict:
    """Session with 2 categories, 4 images each, all labeled."""
    sess = client.post("/platform/sessions", json={"name": "train-test"}).json()
    sid = sess["id"]

    cat_stop = client.post("/platform/categories", json={"session_id": sid, "name": "stop_sign"}).json()["id"]
    cat_speed = client.post("/platform/categories", json={"session_id": sid, "name": "speed_limit"}).json()["id"]

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(exist_ok=True)

    for i in range(4):
        img_bytes = io.BytesIO()
        Image.new("RGB", (64, 64), color=(200, 50, i * 10)).save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        resp = client.post(
            f"/platform/sessions/{sid}/upload",
            files=[("files", (f"stop_{i}.jpg", img_bytes, "image/jpeg"))],
        )
        iid = resp.json()["images"][0]["image_id"]
        client.post("/platform/label", json={"image_id": iid, "category_id": cat_stop, "confirmed": True})

    for i in range(4):
        img_bytes = io.BytesIO()
        Image.new("RGB", (64, 64), color=(50, 50, 200 - i * 10)).save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        resp = client.post(
            f"/platform/sessions/{sid}/upload",
            files=[("files", (f"speed_{i}.jpg", img_bytes, "image/jpeg"))],
        )
        iid = resp.json()["images"][0]["image_id"]
        client.post("/platform/label", json={"image_id": iid, "category_id": cat_speed, "confirmed": True})

    return {"session_id": sid, "cat_stop": cat_stop, "cat_speed": cat_speed}


def wait_for_training(client: TestClient, session_id: int, timeout: int = 60) -> dict:
    """Poll train/status until done or error."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get("/platform/train/status", params={"session_id": session_id}).json()
        if status["status"] in ("done", "error"):
            return status
        time.sleep(0.5)
    raise TimeoutError("Training did not finish in time")


# ------------------------------------------------------------------
# POST /platform/train
# ------------------------------------------------------------------

class TestTrainEndpoint:
    def test_start_training_returns_202(self, client: TestClient, ready_session: dict) -> None:
        resp = client.post("/platform/train", json={
            "session_id": ready_session["session_id"], "epochs": 1
        })
        assert resp.status_code == 202

    def test_start_training_status_running(self, client: TestClient, ready_session: dict) -> None:
        resp = client.post("/platform/train", json={
            "session_id": ready_session["session_id"], "epochs": 2
        })
        assert resp.json()["status"] == "running"

    def test_training_completes(self, client: TestClient, ready_session: dict) -> None:
        client.post("/platform/train", json={
            "session_id": ready_session["session_id"], "epochs": 1
        })
        status = wait_for_training(client, ready_session["session_id"])
        assert status["status"] == "done"

    def test_training_progress_has_metrics(self, client: TestClient, ready_session: dict) -> None:
        client.post("/platform/train", json={
            "session_id": ready_session["session_id"], "epochs": 2
        })
        status = wait_for_training(client, ready_session["session_id"])
        assert status["loss"] is not None
        assert status["accuracy"] is not None
        assert 0.0 <= status["accuracy"] <= 100.0

    def test_duplicate_train_returns_409(self, client: TestClient, ready_session: dict) -> None:
        client.post("/platform/train", json={
            "session_id": ready_session["session_id"], "epochs": 3
        })
        resp2 = client.post("/platform/train", json={
            "session_id": ready_session["session_id"], "epochs": 1
        })
        assert resp2.status_code == 409

    def test_train_missing_session(self, client: TestClient) -> None:
        resp = client.post("/platform/train", json={"session_id": 9999, "epochs": 1})
        assert resp.status_code == 404

    def test_train_no_data_returns_error_status(self, client: TestClient) -> None:
        sess = client.post("/platform/sessions", json={"name": "empty"}).json()
        client.post("/platform/train", json={"session_id": sess["id"], "epochs": 1})
        status = wait_for_training(client, sess["id"])
        assert status["status"] == "error"
        assert status["error"] is not None


# ------------------------------------------------------------------
# GET /platform/train/status
# ------------------------------------------------------------------

class TestTrainStatus:
    def test_status_idle_before_training(self, client: TestClient, ready_session: dict) -> None:
        resp = client.get("/platform/train/status", params={"session_id": ready_session["session_id"]})
        assert resp.status_code == 200
        assert resp.json()["status"] == "idle"

    def test_status_has_epoch_after_done(self, client: TestClient, ready_session: dict) -> None:
        client.post("/platform/train", json={
            "session_id": ready_session["session_id"], "epochs": 1
        })
        status = wait_for_training(client, ready_session["session_id"])
        assert status["epoch"] == 1


# ------------------------------------------------------------------
# POST /platform/export
# ------------------------------------------------------------------

class TestExportEndpoint:
    def test_export_after_training(self, client: TestClient, ready_session: dict) -> None:
        client.post("/platform/train", json={
            "session_id": ready_session["session_id"], "epochs": 1
        })
        wait_for_training(client, ready_session["session_id"])
        resp = client.post("/platform/export", params={"session_id": ready_session["session_id"]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"].endswith(".onnx")
        assert "stop_sign" in data["categories"]
        assert "speed_limit" in data["categories"]

    def test_export_while_training_returns_409(self, client: TestClient, ready_session: dict) -> None:
        client.post("/platform/train", json={
            "session_id": ready_session["session_id"], "epochs": 100
        })
        resp = client.post("/platform/export", params={"session_id": ready_session["session_id"]})
        assert resp.status_code == 409

    def test_export_missing_session(self, client: TestClient) -> None:
        resp = client.post("/platform/export", params={"session_id": 9999})
        assert resp.status_code == 404

    def test_download_onnx_file(self, client: TestClient, ready_session: dict) -> None:
        client.post("/platform/train", json={
            "session_id": ready_session["session_id"], "epochs": 1
        })
        wait_for_training(client, ready_session["session_id"])
        export = client.post("/platform/export", params={"session_id": ready_session["session_id"]}).json()
        filename = export["filename"]
        dl = client.get(f"/platform/download/{filename}")
        assert dl.status_code == 200
        assert dl.headers["content-type"] == "application/octet-stream"

    def test_download_missing_file(self, client: TestClient) -> None:
        resp = client.get("/platform/download/nonexistent.onnx")
        assert resp.status_code == 404