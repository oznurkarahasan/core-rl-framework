# Image upload + preprocessing for the Visual Learning Platform.
#
# Responsibilities:
#   - Accept uploaded image files (bytes)
#   - Validate format (JPEG, PNG, WEBP, BMP)
#   - Resize to target size while preserving aspect ratio (letterbox)
#   - Save to uploads directory with a unique filename
#   - Return metadata for DB registration

import hashlib
import time
from pathlib import Path
from typing import NamedTuple

from PIL import Image, UnidentifiedImageError
import io

# Supported input formats
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Target size for all stored images
TARGET_SIZE = (416, 416)

# Default upload directory
DEFAULT_UPLOAD_DIR = Path("uploads")


class UploadResult(NamedTuple):
    filename: str       # final filename on disk (e.g. "abc123.jpg")
    path: str           # absolute path as string
    original_name: str  # original uploaded filename
    width: int
    height: int


class ImageUploader:
    """
    Handles image upload, validation, resizing, and saving.

    Usage:
        uploader = ImageUploader(upload_dir=Path("uploads"))
        result = uploader.process(file_bytes=b"...", original_filename="photo.jpg")
        image_id = db.add_image(session_id, result.filename, result.path)
    """

    def __init__(
        self,
        upload_dir: Path = DEFAULT_UPLOAD_DIR,
        target_size: tuple[int, int] = TARGET_SIZE,
    ) -> None:
        self.upload_dir = Path(upload_dir)
        self.target_size = target_size
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def process(self, file_bytes: bytes, original_filename: str) -> UploadResult:
        """
        Validate, resize, and save an uploaded image.

        Args:
            file_bytes: Raw bytes of the uploaded file.
            original_filename: Original name from the upload form.

        Returns:
            UploadResult with filename, path, and dimensions.

        Raises:
            ValueError: If the format is not supported or bytes are not a valid image.
        """
        self._validate_extension(original_filename)
        image = self._decode(file_bytes, original_filename)
        image = self._resize(image)
        filename, save_path = self._save(image, original_filename, file_bytes)

        return UploadResult(
            filename=filename,
            path=str(save_path),
            original_name=original_filename,
            width=image.width,
            height=image.height,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_extension(self, filename: str) -> None:
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

    def _decode(self, file_bytes: bytes, filename: str) -> Image.Image:
        try:
            image = Image.open(io.BytesIO(file_bytes))
            image.verify()                   # catches truncated / corrupt files
            image = Image.open(io.BytesIO(file_bytes))  # reopen after verify
            return image.convert("RGB")      # normalise to RGB (strips alpha, CMYK, etc.)
        except UnidentifiedImageError:
            raise ValueError(f"'{filename}' is not a valid image file.")
        except Exception as e:
            raise ValueError(f"Failed to decode '{filename}': {e}")

    def _resize(self, image: Image.Image) -> Image.Image:
        """
        Letterbox resize: fit inside target_size, pad with black to exact size.
        Preserves aspect ratio — no distortion.
        """
        target_w, target_h = self.target_size
        orig_w, orig_h = image.size

        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        resized = image.resize((new_w, new_h), Image.LANCZOS)

        canvas = Image.new("RGB", self.target_size, (0, 0, 0))
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        canvas.paste(resized, (offset_x, offset_y))

        return canvas

    def _save(
        self, image: Image.Image, original_filename: str, original_bytes: bytes
    ) -> tuple[str, Path]:
        """
        Generate a unique filename from content hash + timestamp, save as JPEG.
        Returns (filename, absolute_path).
        """
        content_hash = hashlib.sha256(original_bytes).hexdigest()[:12]
        timestamp = int(time.time() * 1000)
        filename = f"{timestamp}_{content_hash}.jpg"
        save_path = self.upload_dir / filename

        image.save(save_path, format="JPEG", quality=90)
        return filename, save_path.resolve()
