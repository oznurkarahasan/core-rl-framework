import random
import threading
import time

import numpy as np
import uvicorn

from core_rl.active_learning.review_queue import ReviewQueue
from core_rl.api.app import create_app

queue = ReviewQueue(capacity=100)
app = create_app(queue=queue)


def _dummy_feeder():
    """Periodically pushes random uncertain states so the queue has data to browse."""
    while True:
        if len(queue) < 10:
            state = np.array([random.uniform(-1, 1) for _ in range(8)])
            entropy = random.uniform(0.5, 2.0)
            queue.push(state=state, entropy=entropy)
        time.sleep(5)


if __name__ == "__main__":
    threading.Thread(target=_dummy_feeder, daemon=True).start()
    print("API running at http://localhost:8000  (docs: http://localhost:8000/docs)")
    uvicorn.run(app, host="0.0.0.0", port=8000)
