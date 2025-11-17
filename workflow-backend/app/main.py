# ============================================
# FILE: workflow-backend/app/main.py
# ============================================
"""
Workflow Scheduler API
Branch-aware DAG scheduler for whole-slide image processing with InstanSeg
"""

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
import uuid
import asyncio
import logging
import os
from pathlib import Path

# Import scheduler - using try/except for safety
try:
    from app.scheduler.branch_scheduler import scheduler
except ImportError as e:
    logging.error(f"Failed to import scheduler: {e}")
    scheduler = None

# Import Celery app (not the task directly to avoid circular imports)
try:
    from app.celery_app import celery_app
except ImportError as e:
    logging.error(f"Failed to import celery_app: {e}")
    celery_app = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Branch-Aware Workflow Scheduler",
    description="DAG scheduler for whole-slide image segmentation with InstanSeg",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connections for real-time updates
active_connections = {}


# ============================================
# FILE UPLOAD ENDPOINT
# ============================================

@app.post("/upload/{user_id}")
async def upload_file(
    user_id: str,
    file: UploadFile = File(...)
):
    """
    Upload a .svs or image file for processing

    Returns the file path that can be used in workflow submission
    """
    try:
        # Create user-specific upload directory
        upload_dir = Path(f"uploads/{user_id}")
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Save file with original name
        file_path = upload_dir / file.filename

        # Write file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"File uploaded: {file_path} ({len(content)} bytes)")

        return {
            "filename": file.filename,
            "filepath": str(file_path),
            "size": len(content)
        }

    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


# ============================================
# SERVE RESULT IMAGES
# ============================================

@app.get("/outputs/{user_id}/{workflow_name}/{filename}")
async def get_output_file(user_id: str, workflow_name: str, filename: str):
    """
    Serve result images from outputs directory

    Args:
        user_id: User ID
        workflow_name: Workflow name (sanitized)
        filename: Name of the result file (e.g., job_id_mask.png)

    Returns:
        FileResponse with the requested image
    """
    file_path = Path(f"outputs/{user_id}/{workflow_name}/{filename}")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Security check: ensure path is within outputs directory
    if not str(file_path.resolve()).startswith(str(Path("outputs").resolve())):
        raise HTTPException(status_code=403, detail="Access denied")

    return FileResponse(file_path)


# ============================================
# WORKFLOW ENDPOINTS
# ============================================

@app.post("/workflows")
async def create_workflow(
    workflow: dict,
    x_user_id: str = Header(..., description="User ID for multi-tenant isolation")
):
    """
    Create and submit a workflow for WSI segmentation

    Example request body:
    {
      "name": "Cell Segmentation Pipeline",
      "dag": {
        "branch_A": [
          {"job_id": "job1", "image": "slide1.svs", "type": "segment"},
          {"job_id": "job2", "image": "slide2.svs", "type": "segment"}
        ],
        "branch_B": [
          {"job_id": "job3", "image": "slide3.svs", "type": "segment"}
        ]
      }
    }
    """
    # Generate workflow ID
    workflow_id = workflow.get('workflow_id', str(uuid.uuid4()))
    workflow['workflow_id'] = workflow_id

    logger.info(f"Received workflow from user {x_user_id}: {workflow_id}")

    # Submit to scheduler
    result = await scheduler.submit_workflow(x_user_id, workflow)

    return result


@app.get("/workflows/{workflow_id}")
async def get_workflow_status(
    workflow_id: str,
    x_user_id: str = Header(..., description="User ID")
):
    """Get workflow status with progress"""
    workflow = scheduler.get_workflow_status(workflow_id)

    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    # Check user ownership
    if workflow.get('user_id') != x_user_id:
        raise HTTPException(status_code=404, detail="Workflow not found")

    return workflow


@app.get("/workflows")
async def list_workflows(
    x_user_id: str = Header(..., description="User ID")
):
    """List all workflows for a user"""
    user_workflows = [
        w for w in scheduler.workflows.values()
        if w.get('user_id') == x_user_id
    ]
    return {"workflows": user_workflows}


@app.delete("/workflows/{workflow_id}")
async def cancel_workflow(
    workflow_id: str,
    x_user_id: str = Header(...)
):
    """Cancel a workflow (cancels all queued jobs)"""
    workflow = scheduler.get_workflow_status(workflow_id)

    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if workflow.get('user_id') != x_user_id:
        raise HTTPException(status_code=404, detail="Workflow not found")

    # Cancel all queued jobs in the workflow
    cancelled_jobs = []
    for branch_id in workflow.get('branches', []):
        if branch_id in scheduler.branch_queues:
            queue = scheduler.branch_queues[branch_id]
            # Remove jobs belonging to this workflow
            remaining = []
            for job in queue:
                if job.get('workflow_id') == workflow_id:
                    cancelled_jobs.append(job['job_id'])
                else:
                    remaining.append(job)
            scheduler.branch_queues[branch_id] = remaining

    # Update workflow status
    if workflow_id in scheduler.workflows:
        scheduler.workflows[workflow_id]['status'] = 'CANCELLED'

    logger.info(f"Cancelled workflow {workflow_id}: {len(cancelled_jobs)} jobs removed from queue")

    return {
        "message": "Workflow cancelled",
        "workflow_id": workflow_id,
        "cancelled_jobs": len(cancelled_jobs)
    }


@app.delete("/jobs/{job_id}")
async def cancel_job(
    job_id: str,
    x_user_id: str = Header(...)
):
    """Cancel a single job (queued or running)"""
    # Find the job in any branch queue
    job_found = False
    for branch_id, queue in scheduler.branch_queues.items():
        for i, job in enumerate(queue):
            if job['job_id'] == job_id:
                # Check ownership
                if job.get('user_id') != x_user_id:
                    raise HTTPException(status_code=404, detail="Job not found")

                # Remove from queue
                queue.pop(i)
                job_found = True
                logger.info(f"Cancelled queued job {job_id} from branch {branch_id}")
                return {
                    "message": "Job cancelled (was queued)",
                    "job_id": job_id,
                    "branch_id": branch_id,
                    "status": "CANCELLED"
                }

    # Check if job is currently running
    if job_id in scheduler.active_jobs:
        job = scheduler.active_jobs[job_id]
        if job.get('user_id') != x_user_id:
            raise HTTPException(status_code=404, detail="Job not found")

        # Revoke the Celery task
        if job_id in _active_tasks:
            task = _active_tasks[job_id]
            task.revoke(terminate=True)  # Terminate the task
            logger.info(f"Revoked running job {job_id}")

            # Clean up
            await scheduler.job_completed(job_id, 'CANCELLED')
            del _active_tasks[job_id]

            return {
                "message": "Job cancelled (was running)",
                "job_id": job_id,
                "branch_id": job.get('branch_id'),
                "status": "CANCELLED"
            }

    if not job_found:
        raise HTTPException(status_code=404, detail="Job not found or already completed")


# ============================================
# SCHEDULER STATUS ENDPOINTS
# ============================================

@app.get("/scheduler/stats")
async def get_scheduler_stats():
    """Get overall scheduler statistics"""
    return scheduler.get_scheduler_stats()


@app.get("/scheduler/branches")
async def get_branch_status():
    """Get status of all branches"""
    branch_info = []

    for branch_id, queue in scheduler.branch_queues.items():
        active_job = scheduler.branch_active.get(branch_id)

        branch_info.append({
            "branch_id": branch_id,
            "queued_jobs": len(queue),
            "active_job": active_job,
            "status": "BUSY" if active_job else "IDLE"
        })

    return {"branches": branch_info}


# ============================================
# WEBSOCKET FOR REAL-TIME UPDATES
# ============================================

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket for real-time workflow updates"""
    await websocket.accept()
    active_connections[user_id] = websocket

    try:
        while True:
            # Keep connection alive and send updates
            stats = scheduler.get_scheduler_stats()
            await websocket.send_json({
                "type": "stats",
                "data": stats
            })
            await asyncio.sleep(2)

    except WebSocketDisconnect:
        del active_connections[user_id]
        logger.info(f"WebSocket disconnected: {user_id}")


async def broadcast_update(user_id: str, message: dict):
    """Send update to user's WebSocket"""
    if user_id in active_connections:
        try:
            await active_connections[user_id].send_json(message)
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")


# ============================================
# BACKGROUND JOB EXECUTOR
# ============================================

_executor_running = False
_active_tasks = {}  # job_id -> AsyncResult

async def job_executor():
    """
    Background task that continuously dispatches jobs
    Runs the branch-aware scheduling logic
    """
    global _executor_running, _active_tasks

    if _executor_running:
        return  # Already running

    _executor_running = True
    logger.info("Job executor started")

    try:
        while True:
            # Get next executable job
            job = await scheduler.get_next_job()

            if job:
                job_id = job['job_id']
                user_id = job['user_id']

                logger.info(f"Executing job {job_id} for user {user_id}")

                # Submit to Celery worker using task name (avoids circular import)
                task = celery_app.send_task('app.workers.segment_wsi', args=[job])

                # Store task for tracking
                _active_tasks[job_id] = task

                # Notify user via WebSocket
                await broadcast_update(user_id, {
                    "type": "job_started",
                    "job_id": job_id,
                    "workflow_id": job.get('workflow_id')
                })

            # Check for completed tasks
            completed_jobs = []
            for job_id, task in list(_active_tasks.items()):
                if task.ready():
                    completed_jobs.append(job_id)

                    # Get result and update scheduler
                    try:
                        if task.successful():
                            result = task.result
                            logger.info(f"Job {job_id} completed successfully")
                            # Pass the result to the scheduler
                            await scheduler.job_completed(job_id, 'SUCCESS', result)
                        else:
                            logger.error(f"Job {job_id} failed")
                            await scheduler.job_completed(job_id, 'FAILED')
                    except Exception as e:
                        logger.error(f"Error processing completed job {job_id}: {e}")
                        await scheduler.job_completed(job_id, 'FAILED')

            # Remove completed tasks
            for job_id in completed_jobs:
                del _active_tasks[job_id]

            # Wait before next iteration
            await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"Job executor error: {e}", exc_info=True)
        _executor_running = False


# ============================================
# HEALTH & DEBUG ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "service": "Branch-Aware Workflow Scheduler",
        "status": "running",
        "version": "2.0.0",
        "features": [
            "Branch-aware DAG scheduling",
            "Multi-tenant isolation (max 3 users)",
            "InstanSeg whole-slide image segmentation",
            "Smart tiling with 60-70% speedup"
        ],
        "docs": "/docs"
    }


@app.get("/debug/workflows")
async def debug_workflows():
    """Debug: List all workflows"""
    return {"workflows": list(scheduler.workflows.values())}


@app.get("/debug/queues")
async def debug_queues():
    """Debug: Show all branch queues"""
    queues = {}
    for branch_id, queue in scheduler.branch_queues.items():
        queues[branch_id] = [
            {"job_id": j['job_id'], "user_id": j.get('user_id')}
            for j in queue
        ]
    return {"queues": queues}


# ============================================
# STARTUP
# ============================================

@app.on_event("startup")
async def startup_event():
    logger.info("Workflow Scheduler API started")
    logger.info(f"Max workers: {scheduler.max_workers}")
    logger.info(f"Max active users: {scheduler.max_active_users}")
    logger.info("InstanSeg WSI segmentation enabled")

    # Start the job executor background task
    asyncio.create_task(job_executor())
    logger.info("Job executor started")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
