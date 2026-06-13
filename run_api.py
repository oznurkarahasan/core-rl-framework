import uvicorn
from core_rl.active_learning.review_queue import ReviewQueue
from core_rl.api.app import create_app

queue = ReviewQueue(capacity=100)
app = create_app(queue=queue)

if __name__ == "__main__":
    print("API running at http://localhost:8000  (docs: http://localhost:8000/docs)")
    uvicorn.run(app, host="0.0.0.0", port=8000)