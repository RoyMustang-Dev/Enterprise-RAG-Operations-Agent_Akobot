"""
Celery App Bootstrap
"""
import os
from celery import Celery


def create_celery_app() -> Celery:
    broker = os.getenv("CELERY_BROKER_URL") or os.getenv("REDIS_URL")
    backend = os.getenv("CELERY_RESULT_BACKEND") or broker

    app = Celery(
        "enterprise_rag",
        broker=broker,
        backend=backend,
        include=["app.infra.celery_tasks"]
    )
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        worker_prefetch_multiplier=1,
    )
    return app


celery_app = create_celery_app()
