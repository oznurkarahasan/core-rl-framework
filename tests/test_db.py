# tests/test_db.py
#
# Test suite for core_rl.data.db.PlatformDB (Phase 7 — Task 7.1)

import pytest
from pathlib import Path
from core_rl.data.db import PlatformDB


@pytest.fixture
def db(tmp_path: Path) -> PlatformDB:
    """Fresh in-memory-style DB for each test (tmp_path is unique per test)."""
    d = PlatformDB(path=tmp_path / "test.db")
    d.init()
    return d


@pytest.fixture
def session(db: PlatformDB) -> int:
    return db.create_session("test-session")


@pytest.fixture
def populated(db: PlatformDB, session: int) -> dict:
    """DB with one session, two categories, three images."""
    cat1 = db.add_category(session, "stop_sign")
    cat2 = db.add_category(session, "speed_limit")
    img1 = db.add_image(session, "stop1.jpg", "/uploads/stop1.jpg")
    img2 = db.add_image(session, "stop2.jpg", "/uploads/stop2.jpg")
    img3 = db.add_image(session, "speed1.jpg", "/uploads/speed1.jpg")
    return {
        "session_id": session,
        "cat_stop": cat1,
        "cat_speed": cat2,
        "img1": img1,
        "img2": img2,
        "img3": img3,
    }


# ------------------------------------------------------------------
# init
# ------------------------------------------------------------------

class TestInit:
    def test_init_is_idempotent(self, tmp_path: Path) -> None:
        """Calling init() twice must not raise or duplicate tables."""
        db = PlatformDB(path=tmp_path / "idempotent.db")
        db.init()
        db.init()

    def test_foreign_keys_enabled(self, db: PlatformDB) -> None:
        """Inserting an image with a non-existent session_id must fail."""
        import sqlite3
        with pytest.raises(sqlite3.IntegrityError):
            db.add_image(session_id=9999, filename="x.jpg", path="/x.jpg")


# ------------------------------------------------------------------
# Sessions
# ------------------------------------------------------------------

class TestSessions:
    def test_create_session_returns_id(self, db: PlatformDB) -> None:
        sid = db.create_session("my-session")
        assert isinstance(sid, int)
        assert sid > 0

    def test_get_session(self, db: PlatformDB, session: int) -> None:
        row = db.get_session(session)
        assert row is not None
        assert row["name"] == "test-session"

    def test_get_session_missing(self, db: PlatformDB) -> None:
        assert db.get_session(9999) is None

    def test_list_sessions_empty(self, db: PlatformDB) -> None:
        assert db.list_sessions() == []

    def test_list_sessions(self, db: PlatformDB) -> None:
        db.create_session("a")
        db.create_session("b")
        sessions = db.list_sessions()
        assert len(sessions) == 2
        # newest first
        assert sessions[0]["name"] == "b"

    def test_set_checkpoint(self, db: PlatformDB, session: int) -> None:
        db.set_checkpoint(session, "/checkpoints/model_v1.onnx")
        row = db.get_session(session)
        assert row["checkpoint_path"] == "/checkpoints/model_v1.onnx"

    def test_checkpoint_defaults_to_none(self, db: PlatformDB, session: int) -> None:
        row = db.get_session(session)
        assert row["checkpoint_path"] is None


# ------------------------------------------------------------------
# Categories
# ------------------------------------------------------------------

class TestCategories:
    def test_add_category(self, db: PlatformDB, session: int) -> None:
        cid = db.add_category(session, "stop_sign")
        assert isinstance(cid, int)
        assert cid > 0

    def test_add_duplicate_category_returns_same_id(self, db: PlatformDB, session: int) -> None:
        cid1 = db.add_category(session, "stop_sign")
        cid2 = db.add_category(session, "stop_sign")
        assert cid1 == cid2

    def test_same_name_different_sessions(self, db: PlatformDB) -> None:
        s1 = db.create_session("s1")
        s2 = db.create_session("s2")
        cid1 = db.add_category(s1, "stop_sign")
        cid2 = db.add_category(s2, "stop_sign")
        assert cid1 != cid2

    def test_list_categories(self, db: PlatformDB, session: int) -> None:
        db.add_category(session, "zebra")
        db.add_category(session, "apple")
        cats = db.list_categories(session)
        # ordered by name
        assert [c["name"] for c in cats] == ["apple", "zebra"]

    def test_list_categories_empty(self, db: PlatformDB, session: int) -> None:
        assert db.list_categories(session) == []


# ------------------------------------------------------------------
# Images
# ------------------------------------------------------------------

class TestImages:
    def test_add_image(self, db: PlatformDB, session: int) -> None:
        iid = db.add_image(session, "img.jpg", "/uploads/img.jpg")
        assert isinstance(iid, int)
        assert iid > 0

    def test_list_images(self, db: PlatformDB, session: int) -> None:
        db.add_image(session, "a.jpg", "/uploads/a.jpg")
        db.add_image(session, "b.jpg", "/uploads/b.jpg")
        imgs = db.list_images(session)
        assert len(imgs) == 2

    def test_list_images_scoped_to_session(self, db: PlatformDB) -> None:
        s1 = db.create_session("s1")
        s2 = db.create_session("s2")
        db.add_image(s1, "a.jpg", "/a.jpg")
        db.add_image(s2, "b.jpg", "/b.jpg")
        assert len(db.list_images(s1)) == 1
        assert len(db.list_images(s2)) == 1

    def test_count_images(self, db: PlatformDB, session: int) -> None:
        assert db.count_images(session) == 0
        db.add_image(session, "a.jpg", "/a.jpg")
        db.add_image(session, "b.jpg", "/b.jpg")
        assert db.count_images(session) == 2

    def test_get_unlabeled_returns_all_when_no_labels(self, db: PlatformDB, populated: dict) -> None:
        unlabeled = db.get_unlabeled_images(populated["session_id"], populated["cat_stop"])
        assert len(unlabeled) == 3

    def test_get_unlabeled_excludes_labeled(self, db: PlatformDB, populated: dict) -> None:
        db.add_label(populated["img1"], populated["cat_stop"], confirmed=True)
        db.add_label(populated["img2"], populated["cat_stop"], confirmed=True)
        unlabeled = db.get_unlabeled_images(populated["session_id"], populated["cat_stop"])
        assert len(unlabeled) == 1
        assert unlabeled[0]["id"] == populated["img3"]

    def test_get_unlabeled_respects_limit(self, db: PlatformDB, session: int) -> None:
        cat = db.add_category(session, "stop_sign")
        for i in range(10):
            db.add_image(session, f"img{i}.jpg", f"/uploads/img{i}.jpg")
        unlabeled = db.get_unlabeled_images(session, cat, limit=4)
        assert len(unlabeled) == 4

    def test_unlabeled_category_independent(self, db: PlatformDB, populated: dict) -> None:
        """Labeling for cat_stop should not affect unlabeled count for cat_speed."""
        db.add_label(populated["img1"], populated["cat_stop"], confirmed=True)
        unlabeled_speed = db.get_unlabeled_images(populated["session_id"], populated["cat_speed"])
        assert len(unlabeled_speed) == 3


# ------------------------------------------------------------------
# Labels
# ------------------------------------------------------------------

class TestLabels:
    def test_add_label(self, db: PlatformDB, populated: dict) -> None:
        lid = db.add_label(populated["img1"], populated["cat_stop"], confirmed=True)
        assert isinstance(lid, int)

    def test_add_label_upsert(self, db: PlatformDB, populated: dict) -> None:
        """Adding the same label twice updates confirmed instead of raising."""
        db.add_label(populated["img1"], populated["cat_stop"], confirmed=True)
        db.add_label(populated["img1"], populated["cat_stop"], confirmed=False)
        data = db.get_labeled_data(populated["session_id"])
        # confirmed=False means it won't show in labeled data (filter: confirmed=1)
        assert len(data) == 0

    def test_count_labels(self, db: PlatformDB, populated: dict) -> None:
        assert db.count_labels(populated["session_id"]) == 0
        db.add_label(populated["img1"], populated["cat_stop"])
        db.add_label(populated["img2"], populated["cat_stop"])
        assert db.count_labels(populated["session_id"]) == 2

    def test_get_labeled_data_structure(self, db: PlatformDB, populated: dict) -> None:
        db.add_label(populated["img1"], populated["cat_stop"])
        db.add_label(populated["img3"], populated["cat_speed"])
        data = db.get_labeled_data(populated["session_id"])
        assert len(data) == 2
        paths = {row["path"] for row in data}
        assert "/uploads/stop1.jpg" in paths
        assert "/uploads/speed1.jpg" in paths

    def test_get_labeled_data_only_confirmed(self, db: PlatformDB, populated: dict) -> None:
        db.add_label(populated["img1"], populated["cat_stop"], confirmed=True)
        db.add_label(populated["img2"], populated["cat_stop"], confirmed=False)
        data = db.get_labeled_data(populated["session_id"])
        assert len(data) == 1
        assert data[0]["path"] == "/uploads/stop1.jpg"


# ------------------------------------------------------------------
# Stats
# ------------------------------------------------------------------

class TestStats:
    def test_stats_empty_session(self, db: PlatformDB, session: int) -> None:
        stats = db.get_stats(session)
        assert stats["total_images"] == 0
        assert stats["total_labels"] == 0
        assert stats["per_category"] == []

    def test_stats_counts(self, db: PlatformDB, populated: dict) -> None:
        db.add_label(populated["img1"], populated["cat_stop"])
        db.add_label(populated["img2"], populated["cat_stop"])
        db.add_label(populated["img3"], populated["cat_speed"])
        stats = db.get_stats(populated["session_id"])
        assert stats["total_images"] == 3
        assert stats["total_labels"] == 3

    def test_stats_per_category(self, db: PlatformDB, populated: dict) -> None:
        db.add_label(populated["img1"], populated["cat_stop"])
        db.add_label(populated["img2"], populated["cat_stop"])
        db.add_label(populated["img3"], populated["cat_speed"])
        stats = db.get_stats(populated["session_id"])
        per_cat = {r["name"]: r["count"] for r in stats["per_category"]}
        assert per_cat["stop_sign"] == 2
        assert per_cat["speed_limit"] == 1

    def test_stats_category_with_zero_labels(self, db: PlatformDB, populated: dict) -> None:
        """Categories with no labels should still appear in per_category with count 0."""
        stats = db.get_stats(populated["session_id"])
        per_cat = {r["name"]: r["count"] for r in stats["per_category"]}
        assert per_cat["stop_sign"] == 0
        assert per_cat["speed_limit"] == 0