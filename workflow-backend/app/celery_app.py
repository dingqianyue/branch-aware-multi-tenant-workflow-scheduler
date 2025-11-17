# ============================================
# FILE: app/celery_app.py
# ============================================
from celery import Celery

# Initialize Celery
celery_app = Celery(
    "workflow_scheduler",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

# Configuration
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_send_sent_event=True,
    task_time_limit=3600,
    task_soft_time_limit=3000,
)

# Explicitly import tasks so they're registered
from app.workers import segment_wsi  # noqa: F401

# Auto-discover tasks from workers directory
celery_app.autodiscover_tasks(['app.workers'])
