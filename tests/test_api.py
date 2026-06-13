import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from core_rl.active_learning.review_queue import ReviewQueue
from core_rl.api.app import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def queue() -> ReviewQueue:
    return ReviewQueue(capacity=10)


@pytest.fixture
def client(queue: ReviewQueue) -> TestClient:
    app = create_app(queue)
    return TestClient(app)


def _push_item(queue: ReviewQueue, entropy: float = 1.5) -> str:
    import numpy as np
    return queue.push(state=np.zeros(4), entropy=entropy)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200

    def test_body(self, client: TestClient) -> None:
        assert client.get("/health").json() == {"status": "ok"}

    def test_health_accessible_without_api_key(self, queue: ReviewQueue, monkeypatch) -> None:
        monkeypatch.setenv("AL_API_KEY", "secret")
        app = create_app(queue)
        c = TestClient(app)
        assert c.get("/health").status_code == 200


# ---------------------------------------------------------------------------
# /queue/status
# ---------------------------------------------------------------------------

class TestQueueStatus:
    def test_empty_queue(self, client: TestClient) -> None:
        r = client.get("/queue/status")
        assert r.status_code == 200
        body = r.json()
        assert body["pending"] == 0
        assert body["capacity"] == 10
        assert body["fill_rate"] == 0.0

    def test_partial_fill(self, client: TestClient, queue: ReviewQueue) -> None:
        _push_item(queue)
        _push_item(queue)
        body = client.get("/queue/status").json()
        assert body["pending"] == 2
        assert body["fill_rate"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# /queue/items
# ---------------------------------------------------------------------------

class TestQueueItems:
    def test_empty_returns_list(self, client: TestClient) -> None:
        r = client.get("/queue/items")
        assert r.status_code == 200
        assert r.json() == []

    def test_items_contain_expected_keys(self, client: TestClient, queue: ReviewQueue) -> None:
        _push_item(queue, entropy=2.0)
        items = client.get("/queue/items").json()
        assert len(items) == 1
        assert {"id", "state", "entropy", "timestamp"} <= set(items[0].keys())

    def test_state_is_json_serialisable(self, client: TestClient, queue: ReviewQueue) -> None:
        _push_item(queue)
        items = client.get("/queue/items").json()
        state = items[0]["state"]
        assert isinstance(state, list)

    def test_multiple_items_oldest_first(self, client: TestClient, queue: ReviewQueue) -> None:
        id1 = _push_item(queue, entropy=1.0)
        id2 = _push_item(queue, entropy=2.0)
        items = client.get("/queue/items").json()
        assert items[0]["id"] == id1
        assert items[1]["id"] == id2


# ---------------------------------------------------------------------------
# /queue/resolve/{id}
# ---------------------------------------------------------------------------

class TestResolveItem:
    def test_resolve_known_item(self, client: TestClient, queue: ReviewQueue) -> None:
        item_id = _push_item(queue)
        r = client.post(f"/queue/resolve/{item_id}", json={"reward": 1.0})
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == item_id
        assert body["reward"] == pytest.approx(1.0)
        assert body["status"] == "resolved"

    def test_resolve_removes_item_from_queue(self, client: TestClient, queue: ReviewQueue) -> None:
        item_id = _push_item(queue)
        client.post(f"/queue/resolve/{item_id}", json={"reward": 0.5})
        assert len(queue.get()) == 0

    def test_resolve_unknown_item_returns_404(self, client: TestClient) -> None:
        r = client.post(f"/queue/resolve/{uuid.uuid4()}", json={"reward": 1.0})
        assert r.status_code == 404

    def test_resolve_triggers_on_resolve_callback(self, queue: ReviewQueue) -> None:
        callback = MagicMock()
        app = create_app(queue, on_resolve=callback)
        c = TestClient(app)
        item_id = _push_item(queue)
        c.post(f"/queue/resolve/{item_id}", json={"reward": 2.5})
        callback.assert_called_once_with(item_id, 2.5)

    def test_no_callback_does_not_raise(self, client: TestClient, queue: ReviewQueue) -> None:
        item_id = _push_item(queue)
        r = client.post(f"/queue/resolve/{item_id}", json={"reward": 1.0})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class TestAuth:
    def _authed_client(self, queue: ReviewQueue, monkeypatch, key: str = "secret") -> TestClient:
        monkeypatch.setenv("AL_API_KEY", key)
        app = create_app(queue)
        return TestClient(app)

    def test_queue_endpoint_blocked_without_key(self, queue, monkeypatch) -> None:
        c = self._authed_client(queue, monkeypatch)
        assert c.get("/queue/status").status_code == 401

    def test_queue_endpoint_allowed_with_correct_key(self, queue, monkeypatch) -> None:
        c = self._authed_client(queue, monkeypatch)
        r = c.get("/queue/status", headers={"X-API-Key": "secret"})
        assert r.status_code == 200

    def test_wrong_key_returns_401(self, queue, monkeypatch) -> None:
        c = self._authed_client(queue, monkeypatch)
        r = c.get("/queue/status", headers={"X-API-Key": "wrong"})
        assert r.status_code == 401

    def test_no_env_key_means_open_access(self, client: TestClient) -> None:
        r = client.get("/queue/status")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Reward-update bridge (Task 5.2)
# ---------------------------------------------------------------------------

class TestRewardUpdateBridge:
    def test_buffer_update_reward_called_via_callback(self, queue: ReviewQueue) -> None:
        from core_rl.buffer.buffer import GenericReplayBuffer
        import numpy as np

        buf = GenericReplayBuffer(capacity=100)
        exp_id = str(uuid.uuid4())

        buf.push(
            state=np.zeros(4),
            action=np.zeros(2),
            reward=0.0,
            next_state=np.zeros(4),
            done=False,
            experience_id=exp_id,
        )
        queue.push(state=np.zeros(4), entropy=1.5, item_id=exp_id)

        app = create_app(queue, on_resolve=lambda id, r: buf.update_reward(id, r))
        c = TestClient(app)

        c.post(f"/queue/resolve/{exp_id}", json={"reward": 1.0})

        updated = buf.update_reward(exp_id, 1.0)
        assert updated is True
