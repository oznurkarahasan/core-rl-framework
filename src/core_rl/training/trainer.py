# src/core_rl/training/trainer.py
#
# MobileNetV2 fine-tuner for the Visual Learning Platform (Task 7.6).
#
# Features:
#   - Loads labeled data from PlatformDB
#   - Fine-tunes MobileNetV2 (pretrained ImageNet weights)
#   - Resumes from last checkpoint if it exists (continual learning)
#   - Streams progress via a callback — wire to FastAPI SSE or print
#   - Saves checkpoint after every epoch

import json
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

from core_rl.data.db import PlatformDB


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class LabeledImageDataset(Dataset):
    """
    Reads confirmed labels from PlatformDB and serves (image_tensor, label_idx).

    categories: list of category names in a fixed order — this order defines
                the class indices and must be consistent across sessions.
    """

    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, rows: list, categories: list[str]) -> None:
        self.cat_to_idx = {c: i for i, c in enumerate(categories)}
        self.samples: list[tuple[str, int]] = []

        for row in rows:
            cat = row["category"]
            path = row["path"]
            if cat in self.cat_to_idx and Path(path).exists():
                self.samples.append((path, self.cat_to_idx[cat]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.TRANSFORM(image), label


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------

class PlatformTrainer:
    """
    Fine-tunes MobileNetV2 on labeled images from a session.

    Usage:
        trainer = PlatformTrainer(db, session_id, checkpoint_dir=Path("checkpoints"))
        trainer.train(epochs=5, on_progress=print)
        onnx_path = trainer.export_onnx()
    """

    def __init__(
        self,
        db: PlatformDB,
        session_id: int,
        checkpoint_dir: Path = Path("checkpoints"),
    ) -> None:
        self.db = db
        self.session_id = session_id
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.categories: list[str] = []
        self.version: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        epochs: int = 5,
        batch_size: int = 16,
        lr: float = 1e-4,
        on_progress: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """
        Train (or continue training) the model on the session's labeled data.

        Args:
            epochs:      Number of epochs to train.
            batch_size:  Batch size for DataLoader.
            lr:          Learning rate.
            on_progress: Called after each epoch with a progress dict:
                         {"epoch": int, "loss": float, "accuracy": float, "total_epochs": int}

        Returns:
            Final metrics dict.
        """
        rows = self.db.get_labeled_data(self.session_id)
        if not rows:
            raise ValueError("No labeled data found. Label some images first.")

        self.categories = sorted({row["category"] for row in rows})
        num_classes = len(self.categories)

        if num_classes < 2:
            raise ValueError(
                f"Need at least 2 categories to train. Found: {self.categories}. "
                "Add more categories and label images for each."
            )

        dataset = LabeledImageDataset(rows, self.categories)
        if len(dataset) < batch_size:
            batch_size = max(1, len(dataset))

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        self._load_or_build_model(num_classes)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        metrics = {}
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            correct = 0
            total = 0

            for images, labels in loader:
                images = images.to(self.device)
                labels = torch.tensor(labels, dtype=torch.long).to(self.device) \
                    if not isinstance(labels, torch.Tensor) else labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)

            avg_loss = round(total_loss / total, 4) if total else 0.0
            accuracy = round(correct / total * 100, 1) if total > 0 else 0.0

            metrics = {
                "epoch": epoch,
                "total_epochs": epochs,
                "loss": avg_loss,
                "accuracy": accuracy,
                "samples": total,
            }

            self._save_checkpoint(epoch)

            if on_progress:
                on_progress(metrics)

        return metrics

    def load_checkpoint(self) -> "PlatformTrainer":
        """
        Load the latest checkpoint for export or inference.
        Categories and version are read from the checkpoint — not from the DB.
        Raises RuntimeError if no checkpoint exists.
        """
        existing = self._latest_checkpoint()
        if existing is None:
            raise RuntimeError(
                f"No checkpoint found for session {self.session_id}. Train the model first."
            )
        state = torch.load(existing, map_location=self.device, weights_only=False)
        self.categories = state["categories"]
        self.version = state["version"]
        self.model = self._build_model(len(self.categories))
        self.model.load_state_dict(state["model"])
        return self

    def export_onnx(self) -> Path:
        """
        Export the current model to ONNX.
        Returns the path to the exported file.
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Run train() first.")

        self.model.eval()
        dummy = torch.randn(1, 3, 224, 224).to(self.device)
        path = self.checkpoint_dir / f"session_{self.session_id}_v{self.version}.onnx"

        torch.onnx.export(
            self.model,
            (dummy,),
            str(path),
            input_names=["image"],
            output_names=["logits"],
        )

        meta = {
            "version": self.version,
            "categories": self.categories,
            "input_size": [1, 3, 224, 224],
            "session_id": self.session_id,
        }
        meta_path = self.checkpoint_dir / f"session_{self.session_id}_v{self.version}.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        self.db.set_checkpoint(self.session_id, str(path))
        return path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _checkpoint_path(self, version: int) -> Path:
        return self.checkpoint_dir / f"session_{self.session_id}_v{version}.pt"

    def _latest_checkpoint(self) -> Optional[Path]:
        checkpoints = sorted(
            self.checkpoint_dir.glob(f"session_{self.session_id}_v*.pt")
        )
        return checkpoints[-1] if checkpoints else None

    def _load_or_build_model(self, num_classes: int) -> None:
        existing = self._latest_checkpoint()

        if existing:
            state = torch.load(existing, map_location=self.device, weights_only=False)
            saved_classes = state.get("num_classes", num_classes)

            if saved_classes == num_classes:
                # Resume: same number of classes
                self.model = self._build_model(num_classes)
                self.model.load_state_dict(state["model"])
                self.version = state.get("version", 0) + 1
                return
            else:
                # Class count changed — rebuild head, keep backbone
                self.model = self._build_model(num_classes)
                backbone_state = {
                    k: v for k, v in state["model"].items()
                    if not k.startswith("classifier")
                }
                missing, unexpected = self.model.load_state_dict(backbone_state, strict=False)
                self.version = state.get("version", 0) + 1
                return

        # Fresh model
        self.model = self._build_model(num_classes)
        self.version = 1

    def _build_model(self, num_classes: int) -> nn.Module:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model.to(self.device)

    def _save_checkpoint(self, epoch: int) -> None:
        path = self._checkpoint_path(self.version)
        torch.save({
            "model": self.model.state_dict(),
            "categories": self.categories,
            "num_classes": len(self.categories),
            "version": self.version,
            "epoch": epoch,
            "session_id": self.session_id,
        }, path)