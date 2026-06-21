# src/core_rl/data/db.py
#
# SQLite database layer for the Visual Learning Platform (Phase 7).
#
# Tables:
#   sessions   — training sessions (each session has its own model checkpoint)
#   categories — label categories, scoped per session
#   images     — uploaded images, linked to a session
#   labels     — human-confirmed labels for images

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional


# Default DB path — can be overridden via PlatformDB(path=...)
DEFAULT_DB_PATH = Path("platform.db")


class PlatformDB:
    """
    Thin wrapper around sqlite3 for the Visual Learning Platform.

    Usage:
        db = PlatformDB()
        db.init()

        session_id = db.create_session("traffic-signs-v1")
        cat_id     = db.add_category(session_id, "stop_sign")
        image_id   = db.add_image(session_id, "stop1.jpg", "/uploads/stop1.jpg")
        db.add_label(image_id, cat_id, confirmed=True)
    """

    def __init__(self, path: Path = DEFAULT_DB_PATH) -> None:
        self.path = Path(path)

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Create tables if they do not exist. Safe to call multiple times."""
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    name              TEXT NOT NULL,
                    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
                    checkpoint_path   TEXT
                );

                CREATE TABLE IF NOT EXISTS categories (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  INTEGER NOT NULL REFERENCES sessions(id),
                    name        TEXT NOT NULL,
                    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(session_id, name)
                );

                CREATE TABLE IF NOT EXISTS images (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  INTEGER NOT NULL REFERENCES sessions(id),
                    filename    TEXT NOT NULL,
                    path        TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS labels (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id    INTEGER NOT NULL REFERENCES images(id),
                    category_id INTEGER NOT NULL REFERENCES categories(id),
                    confirmed   INTEGER NOT NULL DEFAULT 1,
                    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(image_id, category_id)
                );

                CREATE INDEX IF NOT EXISTS idx_images_session
                    ON images(session_id);

                CREATE INDEX IF NOT EXISTS idx_labels_image
                    ON labels(image_id);

                CREATE INDEX IF NOT EXISTS idx_labels_category
                    ON labels(category_id);
            """)

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def create_session(self, name: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO sessions (name) VALUES (?)", (name,)
            )
            return cur.lastrowid

    def delete_session(self, session_id: int) -> bool:
        """Delete a session and all its data. Returns False if not found."""
        with self._conn() as conn:
            if conn.execute("SELECT id FROM sessions WHERE id = ?", (session_id,)).fetchone() is None:
                return False
            conn.execute(
                "DELETE FROM labels WHERE image_id IN (SELECT id FROM images WHERE session_id = ?)",
                (session_id,),
            )
            conn.execute("DELETE FROM categories WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM images WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return True

    def get_session(self, session_id: int) -> Optional[sqlite3.Row]:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()

    def list_sessions(self) -> list[sqlite3.Row]:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC, id DESC"
            ).fetchall()

    def set_checkpoint(self, session_id: int, checkpoint_path: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE sessions SET checkpoint_path = ? WHERE id = ?",
                (checkpoint_path, session_id),
            )

    # ------------------------------------------------------------------
    # Categories
    # ------------------------------------------------------------------

    def add_category(self, session_id: int, name: str) -> int:
        with self._conn() as conn:
            # DO UPDATE SET name=excluded.name is a no-op; required for RETURNING
            # to fire on conflict (DO NOTHING suppresses RETURNING in SQLite).
            row = conn.execute(
                """
                INSERT INTO categories (session_id, name) VALUES (?, ?)
                ON CONFLICT(session_id, name) DO UPDATE SET name = excluded.name
                RETURNING id
                """,
                (session_id, name),
            ).fetchone()
            return row["id"]

    def list_categories(self, session_id: int) -> list[sqlite3.Row]:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM categories WHERE session_id = ? ORDER BY name",
                (session_id,),
            ).fetchall()

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------

    def add_image(self, session_id: int, filename: str, path: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO images (session_id, filename, path) VALUES (?, ?, ?)",
                (session_id, filename, path),
            )
            return cur.lastrowid

    def list_images(self, session_id: int) -> list[sqlite3.Row]:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM images WHERE session_id = ? ORDER BY uploaded_at",
                (session_id,),
            ).fetchall()

    def get_unlabeled_images(self, session_id: int, category_id: int, limit: int = 9) -> list[sqlite3.Row]:
        """Return images that have not yet been labeled for the given category."""
        with self._conn() as conn:
            return conn.execute(
                """
                SELECT i.* FROM images i
                WHERE i.session_id = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM labels l
                      WHERE l.image_id = i.id AND l.category_id = ?
                  )
                ORDER BY i.uploaded_at
                LIMIT ?
                """,
                (session_id, category_id, limit),
            ).fetchall()

    def count_images(self, session_id: int) -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM images WHERE session_id = ?", (session_id,)
            ).fetchone()[0]

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    def add_label(self, image_id: int, category_id: int, confirmed: bool = True) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO labels (image_id, category_id, confirmed)
                VALUES (?, ?, ?)
                ON CONFLICT(image_id, category_id) DO UPDATE SET confirmed = excluded.confirmed
                """,
                (image_id, category_id, int(confirmed)),
            )
            return cur.lastrowid

    def count_labels(self, session_id: int) -> int:
        with self._conn() as conn:
            return conn.execute(
                """
                SELECT COUNT(*) FROM labels l
                JOIN images i ON l.image_id = i.id
                WHERE i.session_id = ?
                """,
                (session_id,),
            ).fetchone()[0]

    def get_labeled_data(self, session_id: int) -> list[sqlite3.Row]:
        """Return all confirmed positive labels for training."""
        with self._conn() as conn:
            return conn.execute(
                """
                SELECT i.path, c.name as category, l.confirmed
                FROM labels l
                JOIN images i ON l.image_id = i.id
                JOIN categories c ON l.category_id = c.id
                WHERE i.session_id = ? AND l.confirmed = 1
                ORDER BY i.id
                """,
                (session_id,),
            ).fetchall()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self, session_id: int) -> dict:
        with self._conn() as conn:
            total_images = conn.execute(
                "SELECT COUNT(*) FROM images WHERE session_id = ?", (session_id,)
            ).fetchone()[0]

            total_labels = conn.execute(
                """
                SELECT COUNT(*) FROM labels l
                JOIN images i ON l.image_id = i.id
                WHERE i.session_id = ?
                """,
                (session_id,),
            ).fetchone()[0]

            per_category = conn.execute(
                """
                SELECT c.name, COUNT(l.id) as count
                FROM categories c
                LEFT JOIN labels l ON l.category_id = c.id
                WHERE c.session_id = ?
                GROUP BY c.name
                ORDER BY c.name
                """,
                (session_id,),
            ).fetchall()

            return {
                "total_images": total_images,
                "total_labels": total_labels,
                "per_category": [dict(r) for r in per_category],
            }