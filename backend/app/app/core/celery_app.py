from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "worker", broker=settings.BROKER_URL, backend=settings.RESULT_BACKEND
)

celery_app.conf.task_routes = {"app.worker.*": "main-queue"}
celery_app.conf.broker_transport_options = {
    "priority_steps": list(range(10)),
    "queue_order_strategy": "priority",
}
