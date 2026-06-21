# tests/test_platform_router.py
#
# Test suite for core_rl.api.platform_router (Phase 7 — Task 7.3)

import io
import pytest
from pathlib import Path
from PIL import Image
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core_rl.api.platform_router import create_platform_router


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def make_jpeg_bytes(width: int = 100, height: int = 80) -> bytes:
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    app = FastAPI()
    router = create_platform_router(
        db_path=tmp_path / "test.db",
        upload_dir=tmp_path / "uploads",
        target_size=(64, 64),
    )
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def session_id(client: TestClient) -> int:
    resp = client.post("/platform/sessions", json={"name": "test-session"})
    return resp.json()["id"]


@pytest.fixture
def category_id(client: TestClient, session_id: int) -> int:
    resp = client.post("/platform/categories", json={
        "session_id": session_id,
        "name": "stop_sign",
    })
    return resp.json()["id"]


@pytest.fixture
def uploaded_image(client: TestClient, session_id: int) -> dict:
    jpeg = make_jpeg_bytes()
    resp = client.post(
        f"/platform/sessions/{session_id}/upload",
        files=[("files", ("test.jpg", jpeg, "image/jpeg"))],
    )
    return resp.json()["images"][0]


# ------------------------------------------------------------------
# Sessions
# ------------------------------------------------------------------

class TestSessions:
    def test_create_session(self, client: TestClient) -> None:
        resp = client.post("/platform/sessions", json={"name": "my-session"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "my-session"
        assert data["id"] > 0
        assert data["checkpoint_path"] is None

    def test_list_sessions_empty(self, client: TestClient) -> None:
        resp = client.get("/platform/sessions")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_sessions(self, client: TestClient) -> None:
        client.post("/platform/sessions", json={"name": "a"})
        client.post("/platform/sessions", json={"name": "b"})
        resp = client.get("/platform/sessions")
        assert resp.status_code == 200
        names = [s["name"] for s in resp.json()]
        assert "a" in names and "b" in names

    def test_list_sessions_newest_first(self, client: TestClient) -> None:
        client.post("/platform/sessions", json={"name": "first"})
        client.post("/platform/sessions", json={"name": "second"})
        resp = client.get("/platform/sessions")
        names = [s["name"] for s in resp.json()]
        assert names[0] == "second"


# ------------------------------------------------------------------
# Upload
# ------------------------------------------------------------------

class TestUpload:
    def test_upload_single_image(self, client: TestClient, session_id: int) -> None:
        jpeg = make_jpeg_bytes()
        resp = client.post(
            f"/platform/sessions/{session_id}/upload",
            files=[("files", ("photo.jpg", jpeg, "image/jpeg"))],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["uploaded"] == 1
        assert len(data["images"]) == 1
        assert data["images"][0]["original_name"] == "photo.jpg"

    def test_upload_multiple_images(self, client: TestClient, session_id: int) -> None:
        files = [
            ("files", (f"img{i}.jpg", make_jpeg_bytes(50 + i, 50), "image/jpeg"))
            for i in range(3)
        ]
        resp = client.post(f"/platform/sessions/{session_id}/upload", files=files)
        assert resp.status_code == 200
        assert resp.json()["uploaded"] == 3

    def test_upload_invalid_format(self, client: TestClient, session_id: int) -> None:
        resp = client.post(
            f"/platform/sessions/{session_id}/upload",
            files=[("files", ("file.gif", b"GIF89a", "image/gif"))],
        )
        assert resp.status_code == 422

    def test_upload_to_missing_session(self, client: TestClient) -> None:
        jpeg = make_jpeg_bytes()
        resp = client.post(
            "/platform/sessions/9999/upload",
            files=[("files", ("photo.jpg", jpeg, "image/jpeg"))],
        )
        assert resp.status_code == 404

    def test_upload_partial_failure_reports_errors(self, client: TestClient, session_id: int) -> None:
        """Valid + invalid file: only valid saved, errors list populated."""
        files = [
            ("files", ("good.jpg", make_jpeg_bytes(), "image/jpeg")),
            ("files", ("bad.gif", b"GIF89a", "image/gif")),
        ]
        resp = client.post(f"/platform/sessions/{session_id}/upload", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["uploaded"] == 1
        assert len(data["errors"]) == 1
        assert data["errors"][0]["file"] == "bad.gif"

    def test_upload_assigns_image_id(self, client: TestClient, session_id: int) -> None:
        jpeg = make_jpeg_bytes()
        resp = client.post(
            f"/platform/sessions/{session_id}/upload",
            files=[("files", ("photo.jpg", jpeg, "image/jpeg"))],
        )
        image_id = resp.json()["images"][0]["image_id"]
        assert isinstance(image_id, int)
        assert image_id > 0


# ------------------------------------------------------------------
# Serve image
# ------------------------------------------------------------------

class TestServeImage:
    def test_serve_uploaded_image(self, client: TestClient, uploaded_image: dict) -> None:
        filename = uploaded_image["filename"]
        resp = client.get(f"/platform/image/{filename}")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"

    def test_serve_missing_image(self, client: TestClient) -> None:
        resp = client.get("/platform/image/nonexistent.jpg")
        assert resp.status_code == 404

    def test_serve_path_traversal_blocked(self, client: TestClient) -> None:
        resp = client.get("/platform/image/..%2F..%2Fetc%2Fpasswd")
        assert resp.status_code in (403, 404)


# ------------------------------------------------------------------
# Categories
# ------------------------------------------------------------------

class TestCategories:
    def test_add_category(self, client: TestClient, session_id: int) -> None:
        resp = client.post("/platform/categories", json={
            "session_id": session_id, "name": "stop_sign"
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "stop_sign"
        assert data["id"] > 0

    def test_add_duplicate_category_returns_same_id(self, client: TestClient, session_id: int) -> None:
        r1 = client.post("/platform/categories", json={"session_id": session_id, "name": "stop_sign"})
        r2 = client.post("/platform/categories", json={"session_id": session_id, "name": "stop_sign"})
        assert r1.json()["id"] == r2.json()["id"]

    def test_add_category_to_missing_session(self, client: TestClient) -> None:
        resp = client.post("/platform/categories", json={"session_id": 9999, "name": "stop_sign"})
        assert resp.status_code == 404

    def test_add_empty_category_name(self, client: TestClient, session_id: int) -> None:
        resp = client.post("/platform/categories", json={"session_id": session_id, "name": "   "})
        assert resp.status_code == 422

    def test_list_categories(self, client: TestClient, session_id: int) -> None:
        client.post("/platform/categories", json={"session_id": session_id, "name": "stop_sign"})
        client.post("/platform/categories", json={"session_id": session_id, "name": "speed_limit"})
        resp = client.get("/platform/categories", params={"session_id": session_id})
        assert resp.status_code == 200
        names = [c["name"] for c in resp.json()]
        assert "stop_sign" in names
        assert "speed_limit" in names

    def test_list_categories_empty(self, client: TestClient, session_id: int) -> None:
        resp = client.get("/platform/categories", params={"session_id": session_id})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_categories_missing_session(self, client: TestClient) -> None:
        resp = client.get("/platform/categories", params={"session_id": 9999})
        assert resp.status_code == 404


# ------------------------------------------------------------------
# Labeling — /next and /label
# ------------------------------------------------------------------

class TestLabeling:
    def test_next_returns_none_when_no_images(
        self, client: TestClient, session_id: int, category_id: int
    ) -> None:
        resp = client.get("/platform/next", params={
            "session_id": session_id, "category_id": category_id
        })
        assert resp.status_code == 200
        assert resp.json() is None

    def test_next_missing_session_returns_404(self, client: TestClient) -> None:
        resp = client.get("/platform/next", params={"session_id": 9999, "category_id": 1})
        assert resp.status_code == 404

    def test_next_returns_image(
        self, client: TestClient, session_id: int, category_id: int, uploaded_image: dict
    ) -> None:
        resp = client.get("/platform/next", params={
            "session_id": session_id, "category_id": category_id
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["image_id"] == uploaded_image["image_id"]
        assert data["path"].startswith("/platform/image/")

    def test_next_returns_none_after_all_labeled(
        self, client: TestClient, session_id: int, category_id: int, uploaded_image: dict
    ) -> None:
        client.post("/platform/label", json={
            "image_id": uploaded_image["image_id"],
            "category_id": category_id,
            "confirmed": True,
        })
        resp = client.get("/platform/next", params={
            "session_id": session_id, "category_id": category_id
        })
        assert resp.json() is None

    def test_submit_label_confirmed(
        self, client: TestClient, uploaded_image: dict, category_id: int
    ) -> None:
        resp = client.post("/platform/label", json={
            "image_id": uploaded_image["image_id"],
            "category_id": category_id,
            "confirmed": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["confirmed"] is True

    def test_submit_label_rejected(
        self, client: TestClient, uploaded_image: dict, category_id: int
    ) -> None:
        resp = client.post("/platform/label", json={
            "image_id": uploaded_image["image_id"],
            "category_id": category_id,
            "confirmed": False,
        })
        assert resp.status_code == 200
        assert resp.json()["confirmed"] is False

    def test_submit_label_invalid_ids_returns_422(self, client: TestClient) -> None:
        resp = client.post("/platform/label", json={
            "image_id": 9999,
            "category_id": 9999,
            "confirmed": True,
        })
        assert resp.status_code == 422


# ------------------------------------------------------------------
# Stats
# ------------------------------------------------------------------

class TestStats:
    def test_stats_empty(self, client: TestClient, session_id: int) -> None:
        resp = client.get("/platform/stats", params={"session_id": session_id})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_images"] == 0
        assert data["total_labels"] == 0

    def test_stats_after_upload_and_label(
        self, client: TestClient, session_id: int, category_id: int, uploaded_image: dict
    ) -> None:
        client.post("/platform/label", json={
            "image_id": uploaded_image["image_id"],
            "category_id": category_id,
            "confirmed": True,
        })
        resp = client.get("/platform/stats", params={"session_id": session_id})
        data = resp.json()
        assert data["total_images"] == 1
        assert data["total_labels"] == 1

    def test_stats_missing_session(self, client: TestClient) -> None:
        resp = client.get("/platform/stats", params={"session_id": 9999})
        assert resp.status_code == 404