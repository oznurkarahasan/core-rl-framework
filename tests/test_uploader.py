# tests/test_uploader.py
#
# Test suite for core_rl.data.uploader.ImageUploader (Phase 7 — Task 7.2)

import io
import pytest
from pathlib import Path
from PIL import Image

from core_rl.data.uploader import ImageUploader, UploadResult, ALLOWED_EXTENSIONS


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def make_image_bytes(width: int = 100, height: int = 80, fmt: str = "JPEG") -> bytes:
    """Create minimal valid image bytes for testing."""
    img = Image.new("RGB", (width, height), color=(120, 60, 200))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def uploader(tmp_path: Path) -> ImageUploader:
    return ImageUploader(upload_dir=tmp_path / "uploads", target_size=(416, 416))


@pytest.fixture
def jpeg_bytes() -> bytes:
    return make_image_bytes(100, 80, "JPEG")


@pytest.fixture
def png_bytes() -> bytes:
    return make_image_bytes(200, 50, "PNG")


@pytest.fixture
def wide_bytes() -> bytes:
    """Wide image to test letterbox padding."""
    return make_image_bytes(800, 200, "JPEG")


@pytest.fixture
def tall_bytes() -> bytes:
    """Tall image to test letterbox padding."""
    return make_image_bytes(100, 600, "JPEG")


@pytest.fixture
def square_bytes() -> bytes:
    return make_image_bytes(300, 300, "JPEG")


# ------------------------------------------------------------------
# Upload directory
# ------------------------------------------------------------------

class TestUploaderSetup:
    def test_creates_upload_dir(self, tmp_path: Path) -> None:
        upload_dir = tmp_path / "new_dir" / "nested"
        ImageUploader(upload_dir=upload_dir)
        assert upload_dir.exists()

    def test_existing_dir_does_not_raise(self, tmp_path: Path) -> None:
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()
        ImageUploader(upload_dir=upload_dir)  # should not raise


# ------------------------------------------------------------------
# Extension validation
# ------------------------------------------------------------------

class TestExtensionValidation:
    @pytest.mark.parametrize("filename", [
        "photo.jpg", "photo.jpeg", "photo.PNG",
        "photo.webp", "photo.bmp", "PHOTO.JPG",
    ])
    def test_allowed_extensions_pass(self, uploader: ImageUploader, jpeg_bytes: bytes, filename: str) -> None:
        result = uploader.process(jpeg_bytes, filename)
        assert isinstance(result, UploadResult)

    @pytest.mark.parametrize("filename", [
        "photo.gif", "photo.tiff", "photo.svg",
        "photo.pdf", "photo.exe", "photo",
    ])
    def test_disallowed_extensions_raise(self, uploader: ImageUploader, jpeg_bytes: bytes, filename: str) -> None:
        with pytest.raises(ValueError, match="Unsupported file type"):
            uploader.process(jpeg_bytes, filename)


# ------------------------------------------------------------------
# Image decoding
# ------------------------------------------------------------------

class TestImageDecoding:
    def test_valid_jpeg(self, uploader: ImageUploader, jpeg_bytes: bytes) -> None:
        result = uploader.process(jpeg_bytes, "test.jpg")
        assert result.filename.endswith(".jpg")

    def test_valid_png(self, uploader: ImageUploader, png_bytes: bytes) -> None:
        result = uploader.process(png_bytes, "test.png")
        assert result.filename.endswith(".jpg")  # always saved as JPEG

    def test_corrupt_bytes_raise(self, uploader: ImageUploader) -> None:
        with pytest.raises(ValueError):
            uploader.process(b"not an image at all", "fake.jpg")

    def test_empty_bytes_raise(self, uploader: ImageUploader) -> None:
        with pytest.raises(ValueError):
            uploader.process(b"", "empty.jpg")

    def test_rgba_image_converted_to_rgb(self, uploader: ImageUploader, tmp_path: Path) -> None:
        """RGBA PNG (transparency) should be accepted and converted to RGB."""
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = uploader.process(buf.getvalue(), "transparent.png")
        saved = Image.open(result.path)
        assert saved.mode == "RGB"


# ------------------------------------------------------------------
# Resize — output dimensions
# ------------------------------------------------------------------

class TestResize:
    def test_output_is_target_size(self, uploader: ImageUploader, jpeg_bytes: bytes) -> None:
        result = uploader.process(jpeg_bytes, "img.jpg")
        assert result.width == 416
        assert result.height == 416

    def test_wide_image_output_size(self, uploader: ImageUploader, wide_bytes: bytes) -> None:
        result = uploader.process(wide_bytes, "wide.jpg")
        assert result.width == 416
        assert result.height == 416

    def test_tall_image_output_size(self, uploader: ImageUploader, tall_bytes: bytes) -> None:
        result = uploader.process(tall_bytes, "tall.jpg")
        assert result.width == 416
        assert result.height == 416

    def test_square_image_output_size(self, uploader: ImageUploader, square_bytes: bytes) -> None:
        result = uploader.process(square_bytes, "square.jpg")
        assert result.width == 416
        assert result.height == 416

    def test_saved_file_matches_reported_size(self, uploader: ImageUploader, jpeg_bytes: bytes) -> None:
        result = uploader.process(jpeg_bytes, "img.jpg")
        saved = Image.open(result.path)
        assert saved.size == (result.width, result.height)

    def test_letterbox_wide_image_has_black_bars(self, uploader: ImageUploader) -> None:
        """Wide image: black padding should appear on top and bottom."""
        wide = make_image_bytes(800, 100, "JPEG")  # very wide, short
        result = uploader.process(wide, "wide.jpg")
        saved = Image.open(result.path)
        # Top-left corner pixel should be black (padding area)
        top_left = saved.getpixel((0, 0))
        assert top_left == (0, 0, 0)

    def test_custom_target_size(self, tmp_path: Path, jpeg_bytes: bytes) -> None:
        uploader = ImageUploader(upload_dir=tmp_path, target_size=(224, 224))
        result = uploader.process(jpeg_bytes, "img.jpg")
        assert result.width == 224
        assert result.height == 224


# ------------------------------------------------------------------
# File saving
# ------------------------------------------------------------------

class TestFileSaving:
    def test_file_is_saved_to_disk(self, uploader: ImageUploader, jpeg_bytes: bytes) -> None:
        result = uploader.process(jpeg_bytes, "img.jpg")
        assert Path(result.path).exists()

    def test_saved_file_is_valid_jpeg(self, uploader: ImageUploader, jpeg_bytes: bytes) -> None:
        result = uploader.process(jpeg_bytes, "img.jpg")
        img = Image.open(result.path)
        assert img.format == "JPEG"

    def test_filename_ends_with_jpg(self, uploader: ImageUploader, jpeg_bytes: bytes) -> None:
        result = uploader.process(jpeg_bytes, "img.jpg")
        assert result.filename.endswith(".jpg")

    def test_unique_filenames_for_same_content_different_time(
        self, uploader: ImageUploader, jpeg_bytes: bytes
    ) -> None:
        """Two uploads of the same file get different filenames (timestamp differs)."""
        import time
        r1 = uploader.process(jpeg_bytes, "img.jpg")
        time.sleep(0.01)
        r2 = uploader.process(jpeg_bytes, "img.jpg")
        assert r1.filename != r2.filename

    def test_original_name_preserved_in_result(self, uploader: ImageUploader, jpeg_bytes: bytes) -> None:
        result = uploader.process(jpeg_bytes, "my_photo.jpg")
        assert result.original_name == "my_photo.jpg"

    def test_multiple_uploads_all_saved(self, uploader: ImageUploader) -> None:
        results = []
        for i in range(5):
            import time; time.sleep(0.01)
            b = make_image_bytes(50 + i * 10, 50, "JPEG")
            r = uploader.process(b, f"img{i}.jpg")
            results.append(r)
        paths = [r.path for r in results]
        assert len(set(paths)) == 5
        for p in paths:
            assert Path(p).exists()