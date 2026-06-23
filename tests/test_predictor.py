# tests/test_predictor.py
#
# Test suite for Predictor and /platform/test/* endpoints (Task 7.9 & 7.10)

import io
import json
import numpy as np
import pytest
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import models
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core_rl.api.platform_router import create_platform_router
from core_rl.training.trainer import PlatformTrainer
from core_rl.training.predictor import Predictor


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
# Helpers
# ------------------------------------------------------------------

def make_jpeg(path: Path, color=(100, 150, 200)) -> None:
    Image.new("RGB", (64, 64), color=color).save(path, format="JPEG")


def make_jpeg_bytes(color=(100, 150, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=color).save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()


def fake_onnx(checkpoint_dir: Path, session_id: int, categories: list[str]) -> Path:
    """Export a minimal untrained model as ONNX for testing."""
    import torch
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(categories))
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    onnx_path = checkpoint_dir / f"session_{session_id}_v1.onnx"
    torch.onnx.export(model, (dummy,), str(onnx_path),
                      input_names=["image"], output_names=["logits"], dynamo=False)
    meta = {"version": 1, "categories": categories,
            "input_size": [1, 3, 224, 224], "session_id": session_id}
    (checkpoint_dir / f"session_{session_id}_v1.json").write_text(json.dumps(meta))
    return onnx_path


# ------------------------------------------------------------------
# Predictor unit tests
# ------------------------------------------------------------------

class TestPredictor:
    def test_predict_returns_expected_keys(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        img_path = tmp_path / "img.jpg"
        make_jpeg(img_path)
        onnx_path = fake_onnx(ckpt, session_id=1, categories=["dur", "park_yeri"])

        predictor = Predictor(onnx_path=onnx_path, categories=["dur", "park_yeri"])
        result = predictor.predict(str(img_path))

        assert "category" in result
        assert "confidence" in result
        assert "scores" in result

    def test_predict_category_in_categories(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        img_path = tmp_path / "img.jpg"
        make_jpeg(img_path)
        onnx_path = fake_onnx(ckpt, 1, ["dur", "park_yeri"])

        predictor = Predictor(onnx_path, ["dur", "park_yeri"])
        result = predictor.predict(str(img_path))

        assert result["category"] in ["dur", "park_yeri"]

    def test_predict_confidence_in_range(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        img_path = tmp_path / "img.jpg"
        make_jpeg(img_path)
        onnx_path = fake_onnx(ckpt, 1, ["dur", "park_yeri"])

        predictor = Predictor(onnx_path, ["dur", "park_yeri"])
        result = predictor.predict(str(img_path))

        assert 0.0 <= result["confidence"] <= 1.0

    def test_scores_sum_to_one(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        img_path = tmp_path / "img.jpg"
        make_jpeg(img_path)
        onnx_path = fake_onnx(ckpt, 1, ["dur", "park_yeri"])

        predictor = Predictor(onnx_path, ["dur", "park_yeri"])
        result = predictor.predict(str(img_path))

        total = sum(result["scores"].values())
        assert abs(total - 1.0) < 1e-4

    def test_scores_has_all_categories(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        img_path = tmp_path / "img.jpg"
        make_jpeg(img_path)
        onnx_path = fake_onnx(ckpt, 1, ["dur", "park_yeri"])

        predictor = Predictor(onnx_path, ["dur", "park_yeri"])
        result = predictor.predict(str(img_path))

        assert set(result["scores"].keys()) == {"dur", "park_yeri"}

    def test_from_session_no_export_raises(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        with pytest.raises(FileNotFoundError, match="No exported model"):
            Predictor.from_session(session_id=99, checkpoint_dir=ckpt)

    def test_from_session_loads_correctly(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        img_path = tmp_path / "img.jpg"
        make_jpeg(img_path)
        fake_onnx(ckpt, session_id=3, categories=["dur", "park_yeri"])

        predictor = Predictor.from_session(session_id=3, checkpoint_dir=ckpt)
        assert predictor.categories == ["dur", "park_yeri"]


# ------------------------------------------------------------------
# API endpoint tests
# ------------------------------------------------------------------

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
    """Session with 2 categories, 4 images, exported ONNX."""
    sess = client.post("/platform/sessions", json={"name": "test-session"}).json()
    sid = sess["id"]

    cat_dur = client.post("/platform/categories", json={"session_id": sid, "name": "dur"}).json()["id"]
    cat_park = client.post("/platform/categories", json={"session_id": sid, "name": "park_yeri"}).json()["id"]

    image_ids = []
    for i in range(4):
        resp = client.post(
            f"/platform/sessions/{sid}/upload",
            files=[("files", (f"img_{i}.jpg", make_jpeg_bytes(), "image/jpeg"))],
        )
        iid = resp.json()["images"][0]["image_id"]
        image_ids.append(iid)
        cat = cat_dur if i < 2 else cat_park
        client.post("/platform/label", json={"image_id": iid, "category_id": cat, "confirmed": True})

    # Create fake ONNX in checkpoint dir
    ckpt_dir = tmp_path / "checkpoints"
    fake_onnx(ckpt_dir, sid, ["dur", "park_yeri"])

    return {"session_id": sid, "image_ids": image_ids}


class TestTestNext:
    def test_returns_prediction(self, client: TestClient, ready_session: dict) -> None:
        resp = client.get("/platform/test/next", params={"session_id": ready_session["session_id"]})
        assert resp.status_code == 200
        data = resp.json()
        assert data is not None
        assert "category" in data
        assert "confidence" in data
        assert "scores" in data

    def test_returns_none_when_no_export(self, client: TestClient) -> None:
        sess = client.post("/platform/sessions", json={"name": "no-export"}).json()
        resp = client.get("/platform/test/next", params={"session_id": sess["id"]})
        assert resp.status_code == 422

    def test_missing_session(self, client: TestClient) -> None:
        resp = client.get("/platform/test/next", params={"session_id": 9999})
        assert resp.status_code == 404

    def test_skips_validated_images(self, client: TestClient, ready_session: dict) -> None:
        sid = ready_session["session_id"]
        # Get first image
        first = client.get("/platform/test/next", params={"session_id": sid}).json()
        # Validate it
        client.post("/platform/test/validate", params={"session_id": sid}, json={
            "image_id": first["image_id"],
            "predicted_category": first["category"],
            "correct": True,
        })
        # Next should be different
        second = client.get("/platform/test/next", params={"session_id": sid}).json()
        if second is not None:
            assert second["image_id"] != first["image_id"]


class TestValidation:
    def test_submit_correct(self, client: TestClient, ready_session: dict) -> None:
        sid = ready_session["session_id"]
        first = client.get("/platform/test/next", params={"session_id": sid}).json()
        resp = client.post("/platform/test/validate", params={"session_id": sid}, json={
            "image_id": first["image_id"],
            "predicted_category": first["category"],
            "correct": True,
        })
        assert resp.status_code == 200
        assert resp.json()["recorded"] is True

    def test_stats_empty(self, client: TestClient, ready_session: dict) -> None:
        sid = ready_session["session_id"]
        resp = client.get("/platform/test/stats", params={"session_id": sid})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["accuracy"] == 0.0

    def test_stats_after_validation(self, client: TestClient, ready_session: dict) -> None:
        sid = ready_session["session_id"]
        first = client.get("/platform/test/next", params={"session_id": sid}).json()
        client.post("/platform/test/validate", params={"session_id": sid}, json={
            "image_id": first["image_id"],
            "predicted_category": first["category"],
            "correct": True,
        })
        stats = client.get("/platform/test/stats", params={"session_id": sid}).json()
        assert stats["total"] == 1
        assert stats["correct"] == 1
        assert stats["accuracy"] == 100.0

    def test_stats_with_wrong_answer(self, client: TestClient, ready_session: dict) -> None:
        sid = ready_session["session_id"]
        first = client.get("/platform/test/next", params={"session_id": sid}).json()
        client.post("/platform/test/validate", params={"session_id": sid}, json={
            "image_id": first["image_id"],
            "predicted_category": first["category"],
            "correct": False,
        })
        stats = client.get("/platform/test/stats", params={"session_id": sid}).json()
        assert stats["total"] == 1
        assert stats["correct"] == 0
        assert stats["accuracy"] == 0.0

    def test_per_category_stats(self, client: TestClient, ready_session: dict) -> None:
        sid = ready_session["session_id"]
        first = client.get("/platform/test/next", params={"session_id": sid}).json()
        client.post("/platform/test/validate", params={"session_id": sid}, json={
            "image_id": first["image_id"],
            "predicted_category": first["category"],
            "correct": True,
        })
        stats = client.get("/platform/test/stats", params={"session_id": sid}).json()
        assert first["category"] in stats["per_category"]