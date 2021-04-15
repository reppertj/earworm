from celery import Celery

celery_app = Celery(
    "worker", broker="redis://queue:6379/0", backend="redis://queue:6379/0"
)

celery_app.conf.task_routes = {"app.worker.*": "main-queue"}
celery_app.conf.broker_transport_options = {
    "priority_steps": list(range(10)),
    "queue_order_strategy": "priority",
}
