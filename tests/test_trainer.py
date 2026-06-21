# tests/test_trainer.py
#
# Test suite for core_rl.training.trainer.PlatformTrainer (Phase 7 — Task 7.6)

import json
import pytest
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import models

from core_rl.data.db import PlatformDB
from core_rl.training.trainer import PlatformTrainer, LabeledImageDataset


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

def make_jpeg(path: Path, width: int = 64, height: int = 64, color=(120, 60, 200)) -> None:
    img = Image.new("RGB", (width, height), color=color)
    img.save(path, format="JPEG")


@pytest.fixture
def db(tmp_path: Path) -> PlatformDB:
    d = PlatformDB(path=tmp_path / "test.db")
    d.init()
    return d


@pytest.fixture
def upload_dir(tmp_path: Path) -> Path:
    d = tmp_path / "uploads"
    d.mkdir()
    return d


@pytest.fixture
def checkpoint_dir(tmp_path: Path) -> Path:
    d = tmp_path / "checkpoints"
    d.mkdir()
    return d


@pytest.fixture
def session_with_data(db: PlatformDB, upload_dir: Path) -> dict:
    sid = db.create_session("test-session")
    cat_stop = db.add_category(sid, "stop_sign")
    cat_speed = db.add_category(sid, "speed_limit")

    for i in range(3):
        path = upload_dir / f"stop_{i}.jpg"
        make_jpeg(path, color=(200, 50, 50))
        iid = db.add_image(sid, path.name, str(path))
        db.add_label(iid, cat_stop, confirmed=True)

    for i in range(3):
        path = upload_dir / f"speed_{i}.jpg"
        make_jpeg(path, color=(50, 50, 200))
        iid = db.add_image(sid, path.name, str(path))
        db.add_label(iid, cat_speed, confirmed=True)

    return {"session_id": sid, "cat_stop": cat_stop, "cat_speed": cat_speed}


@pytest.fixture
def trainer(db: PlatformDB, checkpoint_dir: Path, session_with_data: dict) -> PlatformTrainer:
    return PlatformTrainer(db=db, session_id=session_with_data["session_id"], checkpoint_dir=checkpoint_dir)


# ------------------------------------------------------------------
# LabeledImageDataset
# ------------------------------------------------------------------

class TestLabeledImageDataset:
    def test_len(self, session_with_data: dict, db: PlatformDB) -> None:
        rows = db.get_labeled_data(session_with_data["session_id"])
        ds = LabeledImageDataset(rows, ["speed_limit", "stop_sign"])
        assert len(ds) == 6

    def test_category_to_index(self, session_with_data: dict, db: PlatformDB) -> None:
        rows = db.get_labeled_data(session_with_data["session_id"])
        ds = LabeledImageDataset(rows, ["speed_limit", "stop_sign"])
        _, label = ds[0]
        assert label in [0, 1]

    def test_missing_file_skipped(self, session_with_data: dict, db: PlatformDB, tmp_path: Path) -> None:
        sid = session_with_data["session_id"]
        cat = session_with_data["cat_stop"]
        iid = db.add_image(sid, "ghost.jpg", str(tmp_path / "ghost.jpg"))
        db.add_label(iid, cat, confirmed=True)
        rows = db.get_labeled_data(sid)
        ds = LabeledImageDataset(rows, ["speed_limit", "stop_sign"])
        assert len(ds) == 6

    def test_image_tensor_shape(self, session_with_data: dict, db: PlatformDB) -> None:
        import torch
        rows = db.get_labeled_data(session_with_data["session_id"])
        ds = LabeledImageDataset(rows, ["speed_limit", "stop_sign"])
        tensor, _ = ds[0]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

class TestTrainerValidation:
    def test_no_data_raises(self, db: PlatformDB, checkpoint_dir: Path) -> None:
        sid = db.create_session("empty")
        t = PlatformTrainer(db, sid, checkpoint_dir)
        with pytest.raises(ValueError, match="No labeled data"):
            t.train(epochs=1)

    def test_single_category_raises(self, db: PlatformDB, checkpoint_dir: Path, upload_dir: Path) -> None:
        sid = db.create_session("single-cat")
        cat = db.add_category(sid, "stop_sign")
        path = upload_dir / "img.jpg"
        make_jpeg(path)
        iid = db.add_image(sid, "img.jpg", str(path))
        db.add_label(iid, cat, confirmed=True)
        t = PlatformTrainer(db, sid, checkpoint_dir)
        with pytest.raises(ValueError, match="at least 2 categories"):
            t.train(epochs=1)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

class TestTrainerTraining:
    def test_train_returns_metrics(self, trainer: PlatformTrainer) -> None:
        metrics = trainer.train(epochs=1, batch_size=4)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert metrics["epoch"] == 1

    def test_accuracy_in_range(self, trainer: PlatformTrainer) -> None:
        metrics = trainer.train(epochs=1, batch_size=4)
        assert 0.0 <= metrics["accuracy"] <= 100.0

    def test_on_progress_called(self, trainer: PlatformTrainer) -> None:
        calls = []
        trainer.train(epochs=3, batch_size=4, on_progress=calls.append)
        assert len(calls) == 3
        assert calls[0]["epoch"] == 1
        assert calls[2]["epoch"] == 3

    def test_progress_total_epochs(self, trainer: PlatformTrainer) -> None:
        calls = []
        trainer.train(epochs=2, batch_size=4, on_progress=calls.append)
        for c in calls:
            assert c["total_epochs"] == 2

    def test_checkpoint_saved(self, trainer: PlatformTrainer, checkpoint_dir: Path, session_with_data: dict) -> None:
        trainer.train(epochs=1, batch_size=4)
        checkpoints = list(checkpoint_dir.glob(f"session_{session_with_data['session_id']}_v*.pt"))
        assert len(checkpoints) >= 1

    def test_categories_set_after_train(self, trainer: PlatformTrainer) -> None:
        trainer.train(epochs=1, batch_size=4)
        assert "stop_sign" in trainer.categories
        assert "speed_limit" in trainer.categories


# ------------------------------------------------------------------
# Resume
# ------------------------------------------------------------------

class TestTrainerResume:
    def test_resume_increments_version(self, db: PlatformDB, checkpoint_dir: Path, session_with_data: dict) -> None:
        sid = session_with_data["session_id"]
        t1 = PlatformTrainer(db, sid, checkpoint_dir)
        t1.train(epochs=1, batch_size=4)
        v1 = t1.version
        t2 = PlatformTrainer(db, sid, checkpoint_dir)
        t2.train(epochs=1, batch_size=4)
        assert t2.version == v1 + 1

    def test_resume_produces_valid_metrics(self, db: PlatformDB, checkpoint_dir: Path, session_with_data: dict) -> None:
        sid = session_with_data["session_id"]
        PlatformTrainer(db, sid, checkpoint_dir).train(epochs=1, batch_size=4)
        metrics = PlatformTrainer(db, sid, checkpoint_dir).train(epochs=1, batch_size=4)
        assert "loss" in metrics
        assert metrics["accuracy"] >= 0


# ------------------------------------------------------------------
# ONNX export
# ------------------------------------------------------------------

class TestTrainerExport:
    def test_export_creates_onnx_file(self, trainer: PlatformTrainer, checkpoint_dir: Path) -> None:
        trainer.train(epochs=1, batch_size=4)
        path = trainer.export_onnx()
        assert path.exists()
        assert path.suffix == ".onnx"

    def test_export_creates_meta_json(self, trainer: PlatformTrainer, checkpoint_dir: Path) -> None:
        trainer.train(epochs=1, batch_size=4)
        trainer.export_onnx()
        meta_files = list(checkpoint_dir.glob("*.json"))
        assert len(meta_files) == 1
        meta = json.loads(meta_files[0].read_text())
        assert "categories" in meta
        assert "input_size" in meta

    def test_export_meta_categories(self, trainer: PlatformTrainer, checkpoint_dir: Path) -> None:
        trainer.train(epochs=1, batch_size=4)
        trainer.export_onnx()
        meta = json.loads(list(checkpoint_dir.glob("*.json"))[0].read_text())
        assert set(meta["categories"]) == {"stop_sign", "speed_limit"}

    def test_export_updates_db_checkpoint(self, trainer: PlatformTrainer, db: PlatformDB, session_with_data: dict) -> None:
        trainer.train(epochs=1, batch_size=4)
        path = trainer.export_onnx()
        session = db.get_session(session_with_data["session_id"])
        assert session["checkpoint_path"] == str(path)

    def test_export_without_train_raises(self, db: PlatformDB, checkpoint_dir: Path, session_with_data: dict) -> None:
        t = PlatformTrainer(db, session_with_data["session_id"], checkpoint_dir)
        with pytest.raises(RuntimeError, match="No model loaded"):
            t.export_onnx()