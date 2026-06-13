# tests/test_review_queue.py

import threading
import time
import pytest
import numpy as np
from core_rl.active_learning.review_queue import ReviewQueue


# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------

class TestReviewQueueInit:
    def test_starts_empty(self):
        q = ReviewQueue()
        assert len(q) == 0

    def test_custom_capacity_stored(self):
        q = ReviewQueue(capacity=50)
        assert q._queue.maxlen == 50

    def test_default_capacity(self):
        q = ReviewQueue()
        assert q._queue.maxlen == 1000


# ---------------------------------------------------------
# push()
# ---------------------------------------------------------

class TestReviewQueuePush:
    def test_returns_string_id(self):
        q = ReviewQueue()
        item_id = q.push(np.array([1.0, 2.0]), entropy=0.9)
        assert isinstance(item_id, str)
        assert len(item_id) > 0

    def test_each_push_returns_unique_id(self):
        q = ReviewQueue()
        ids = [q.push(np.array([float(i)]), entropy=0.9) for i in range(10)]
        assert len(set(ids)) == 10

    def test_push_increments_length(self):
        q = ReviewQueue()
        q.push(np.array([1.0]), entropy=0.8)
        assert len(q) == 1

    def test_push_multiple(self):
        q = ReviewQueue()
        for i in range(5):
            q.push(np.array([float(i)]), entropy=float(i) * 0.1)
        assert len(q) == 5

    def test_push_stores_entropy(self):
        q = ReviewQueue()
        q.push(np.array([1.0]), entropy=1.234)
        item = q.get()[0]
        assert item["entropy"] == pytest.approx(1.234)

    def test_push_stores_timestamp(self):
        before = time.time()
        q = ReviewQueue()
        q.push(np.array([1.0]), entropy=0.9)
        after = time.time()
        item = q.get()[0]
        assert before <= item["timestamp"] <= after

    def test_push_stores_state(self):
        q = ReviewQueue()
        state = np.array([3.0, 4.0, 5.0])
        q.push(state, entropy=0.9)
        item = q.get()[0]
        np.testing.assert_array_equal(item["state"], state)

    def test_push_stores_id_in_item(self):
        q = ReviewQueue()
        returned_id = q.push(np.array([1.0]), entropy=0.9)
        item = q.get()[0]
        assert item["id"] == returned_id


# ---------------------------------------------------------
# Capacity eviction — oldest dropped, newest kept
# ---------------------------------------------------------

class TestCapacityEviction:
    def test_does_not_exceed_capacity(self):
        q = ReviewQueue(capacity=3)
        for i in range(10):
            q.push(np.array([float(i)]), entropy=0.9)
        assert len(q) == 3

    def test_oldest_item_evicted(self):
        q = ReviewQueue(capacity=3)
        first_id = q.push(np.array([0.0]), entropy=0.9)
        for i in range(1, 5):
            q.push(np.array([float(i)]), entropy=0.9)
        ids_in_queue = {item["id"] for item in q.get()}
        assert first_id not in ids_in_queue

    def test_newest_item_retained(self):
        q = ReviewQueue(capacity=3)
        for i in range(4):
            last_id = q.push(np.array([float(i)]), entropy=0.9)
        ids_in_queue = {item["id"] for item in q.get()}
        assert last_id in ids_in_queue

    def test_evicted_id_removed_from_index(self):
        q = ReviewQueue(capacity=2)
        old_id = q.push(np.array([0.0]), entropy=0.9)
        q.push(np.array([1.0]), entropy=0.9)
        q.push(np.array([2.0]), entropy=0.9)  # evicts old_id
        assert q.resolve(old_id, reward=1.0) is None


# ---------------------------------------------------------
# get()
# ---------------------------------------------------------

class TestReviewQueueGet:
    def test_returns_list(self):
        q = ReviewQueue()
        assert isinstance(q.get(), list)

    def test_empty_queue_returns_empty_list(self):
        q = ReviewQueue()
        assert q.get() == []

    def test_get_does_not_consume_items(self):
        q = ReviewQueue()
        q.push(np.array([1.0]), entropy=0.9)
        q.get()
        assert len(q) == 1

    def test_get_returns_oldest_first(self):
        q = ReviewQueue()
        ids = [q.push(np.array([float(i)]), entropy=0.9) for i in range(3)]
        items = q.get()
        assert [item["id"] for item in items] == ids

    def test_get_returns_correct_count(self):
        q = ReviewQueue()
        for _ in range(5):
            q.push(np.array([1.0]), entropy=0.9)
        assert len(q.get()) == 5


# ---------------------------------------------------------
# resolve()
# ---------------------------------------------------------

class TestReviewQueueResolve:
    def test_returns_dict_on_valid_id(self):
        q = ReviewQueue()
        item_id = q.push(np.array([1.0]), entropy=0.9)
        result = q.resolve(item_id, reward=1.0)
        assert isinstance(result, dict)

    def test_resolved_item_has_human_reward(self):
        q = ReviewQueue()
        item_id = q.push(np.array([1.0]), entropy=0.9)
        result = q.resolve(item_id, reward=2.5)
        assert result["human_reward"] == pytest.approx(2.5)

    def test_resolved_item_preserves_original_fields(self):
        q = ReviewQueue()
        state = np.array([1.0, 2.0])
        item_id = q.push(state, entropy=1.1)
        result = q.resolve(item_id, reward=1.0)
        assert result["id"] == item_id
        assert result["entropy"] == pytest.approx(1.1)
        np.testing.assert_array_equal(result["state"], state)

    def test_resolve_removes_item_from_queue(self):
        q = ReviewQueue()
        item_id = q.push(np.array([1.0]), entropy=0.9)
        q.resolve(item_id, reward=1.0)
        assert len(q) == 0

    def test_resolve_unknown_id_returns_none(self):
        q = ReviewQueue()
        result = q.resolve("nonexistent-id", reward=1.0)
        assert result is None

    def test_double_resolve_returns_none_second_time(self):
        q = ReviewQueue()
        item_id = q.push(np.array([1.0]), entropy=0.9)
        q.resolve(item_id, reward=1.0)
        result = q.resolve(item_id, reward=2.0)
        assert result is None

    def test_resolve_only_removes_target_item(self):
        q = ReviewQueue()
        id_a = q.push(np.array([1.0]), entropy=0.9)
        id_b = q.push(np.array([2.0]), entropy=0.8)
        q.resolve(id_a, reward=1.0)
        assert len(q) == 1
        remaining_ids = {item["id"] for item in q.get()}
        assert id_b in remaining_ids

    def test_negative_reward_accepted(self):
        q = ReviewQueue()
        item_id = q.push(np.array([1.0]), entropy=0.9)
        result = q.resolve(item_id, reward=-1.0)
        assert result["human_reward"] == pytest.approx(-1.0)


# ---------------------------------------------------------
# Thread safety — concurrent push and resolve
# ---------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_pushes_do_not_raise(self):
        q = ReviewQueue(capacity=500)
        errors = []

        def push_many():
            try:
                for i in range(100):
                    q.push(np.array([float(i)]), entropy=float(i) * 0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=push_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(q) <= 500

    def test_push_and_resolve_concurrent(self):
        q = ReviewQueue(capacity=200)
        pushed_ids = []
        resolved = []
        lock = threading.Lock()

        def producer():
            for i in range(50):
                item_id = q.push(np.array([float(i)]), entropy=0.9)
                with lock:
                    pushed_ids.append(item_id)

        def consumer():
            time.sleep(0.01)
            with lock:
                ids = list(pushed_ids)
            for item_id in ids:
                result = q.resolve(item_id, reward=1.0)
                if result is not None:
                    resolved.append(result)

        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=consumer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # No item should appear twice in resolved
        resolved_ids = [r["id"] for r in resolved]
        assert len(resolved_ids) == len(set(resolved_ids))
