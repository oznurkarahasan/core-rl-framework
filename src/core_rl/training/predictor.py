# src/core_rl/training/predictor.py
#
# ONNX inference engine for the Visual Learning Platform (Task 7.9).
#
# Loads a session's exported ONNX model and runs predictions on images.

import json
import numpy as np
from pathlib import Path
from typing import Optional

from PIL import Image


class Predictor:
    """
    Runs inference on a session's exported ONNX model.

    Usage:
        predictor = Predictor.from_session(session_id=3, checkpoint_dir=Path("checkpoints"))
        result = predictor.predict("/uploads/photo.jpg")
        # {"category": "dur", "confidence": 0.94, "scores": {"dur": 0.94, "park_yeri": 0.06}}
    """

    INPUT_SIZE = (224, 224)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, onnx_path: Path, categories: list[str]) -> None:
        import onnxruntime as ort
        self.categories = categories
        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    @classmethod
    def from_session(cls, session_id: int, checkpoint_dir: Path) -> "Predictor":
        """
        Load the latest exported ONNX model for a session.
        Reads categories from the companion .json metadata file.
        Raises FileNotFoundError if no export exists yet.
        """
        checkpoint_dir = Path(checkpoint_dir)

        # Find latest ONNX for this session — sort numerically to avoid v10 < v9
        def _ver(p: Path) -> int:
            try:
                return int(p.stem.rsplit("_v", 1)[-1])
            except ValueError:
                return 0

        onnx_files = sorted(checkpoint_dir.glob(f"session_{session_id}_v*.onnx"), key=_ver)
        if not onnx_files:
            raise FileNotFoundError(
                f"No exported model found for session {session_id}. "
                "Export the model first via POST /platform/export."
            )
        onnx_path = onnx_files[-1]

        # Load categories from metadata JSON
        meta_path = onnx_path.with_suffix(".json")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        meta = json.loads(meta_path.read_text())
        categories = meta["categories"]

        return cls(onnx_path=onnx_path, categories=categories)

    def predict(self, image_path: str) -> dict:
        """Run inference on an image file path."""
        img = Image.open(image_path).convert("RGB")
        return self._infer(img)

    def predict_bytes(self, data: bytes) -> dict:
        """Run inference on raw image bytes (e.g. from an uploaded file)."""
        from io import BytesIO
        img = Image.open(BytesIO(data)).convert("RGB")
        return self._infer(img)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer(self, img: Image.Image) -> dict:
        tensor = self._preprocess_pil(img)
        logits = self.session.run(None, {self.input_name: tensor})[0][0]
        probs  = self._softmax(logits)
        top_idx = int(np.argmax(probs))
        scores  = {cat: round(float(probs[i]), 4) for i, cat in enumerate(self.categories)}
        return {
            "category":   self.categories[top_idx],
            "confidence": round(float(probs[top_idx]), 4),
            "scores":     scores,
        }

    def _preprocess_pil(self, img: Image.Image) -> np.ndarray:
        img = img.resize(self.INPUT_SIZE, Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - self.MEAN) / self.STD
        arr = arr.transpose(2, 0, 1)
        return arr[np.newaxis, ...]

    def _preprocess(self, image_path: str) -> np.ndarray:
        return self._preprocess_pil(Image.open(image_path).convert("RGB"))

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        e = np.exp(logits - np.max(logits))
        return e / e.sum()