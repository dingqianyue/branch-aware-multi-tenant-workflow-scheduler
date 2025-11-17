# Import tasks so Celery can discover them
from app.workers.segment_worker import segment_wsi

__all__ = ['segment_wsi']
