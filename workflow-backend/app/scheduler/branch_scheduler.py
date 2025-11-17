# ============================================
# FILE: workflow-backend/app/scheduler/branch_scheduler.py
# ============================================
"""
Branch-Aware DAG Scheduler
Key Innovation: Serial within branch, parallel across branches
"""

from collections import defaultdict, deque
from typing import Dict, List, Optional, Set
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class BranchAwareScheduler:
    """
    Scheduler that enforces:
    1. Serial execution within same branch (FIFO)
    2. Parallel execution across different branches
    3. Global worker limit (max concurrent jobs)
    4. User limit (max 3 active users)
    """
    
    def __init__(self, max_workers: int = 10, max_active_users: int = 3):
        self.max_workers = max_workers
        self.max_active_users = max_active_users
        
        # Branch queues: branch_id -> deque of jobs
        self.branch_queues: Dict[str, deque] = defaultdict(deque)
        
        # Active job per branch (only 1 running per branch)
        self.branch_active: Dict[str, str] = {}  # branch_id -> job_id
        
        # All active jobs
        self.active_jobs: Dict[str, dict] = {}  # job_id -> job_data
        
        # User management
        self.active_users: Set[str] = set()
        self.user_queue: deque = deque()  # Waiting users
        self.user_jobs: Dict[str, Set[str]] = defaultdict(set)  # user_id -> set of job_ids
        
        # Workflow tracking
        self.workflows: Dict[str, dict] = {}  # workflow_id -> workflow_data
        
        logger.info(f"Scheduler initialized: max_workers={max_workers}, max_users={max_active_users}")
    
    
    async def submit_workflow(self, user_id: str, workflow_data: dict) -> dict:
        """
        Submit a workflow with multiple jobs organized in branches

        Args:
            user_id: User identifier
            workflow_data: {
                "workflow_id": "wf_123",
                "name": "Cell Segmentation Pipeline",
                "dag": {
                    "branch_A": [job1, job2, job3],  # Sequential
                    "branch_B": [job4, job5]         # Sequential, parallel to A
                }
            }

        Returns:
            Status dict with workflow_id and acceptance status
        """
        workflow_id = workflow_data['workflow_id']
        dag = workflow_data['dag']

        logger.info(f"User {user_id} submitting workflow {workflow_id}")

        # Store workflow first (always store it)
        self.workflows[workflow_id] = {
            "workflow_id": workflow_id,
            "user_id": user_id,
            "name": workflow_data['name'],
            "dag": dag,
            "status": "QUEUED",  # Start as QUEUED, will change to RUNNING when jobs are dispatched
            "created_at": datetime.utcnow().isoformat(),
            "branches": list(dag.keys()),
            "total_jobs": sum(len(jobs) for jobs in dag.values())
        }

        # Check user slot
        if user_id not in self.active_users:
            if len(self.active_users) >= self.max_active_users:
                # User must wait - workflow is stored but jobs not queued yet
                if user_id not in self.user_queue:
                    self.user_queue.append(user_id)
                position = list(self.user_queue).index(user_id) + 1
                logger.info(f"User {user_id} queued at position {position}")
                return {
                    "status": "QUEUED",
                    "message": f"Waiting for available slot (position {position}/{len(self.user_queue)})",
                    "workflow_id": workflow_id,
                    "queue_position": position
                }
            else:
                # Give user a slot
                self.active_users.add(user_id)
                logger.info(f"User {user_id} acquired slot ({len(self.active_users)}/{self.max_active_users})")

        # User is active - enqueue jobs
        return await self._enqueue_workflow_jobs(workflow_id)


    async def _enqueue_workflow_jobs(self, workflow_id: str) -> dict:
        """
        Internal method to enqueue jobs for a workflow.
        Called when workflow is submitted by active user or when user is promoted from queue.
        """
        if workflow_id not in self.workflows:
            logger.error(f"Workflow {workflow_id} not found")
            return {"status": "ERROR", "message": "Workflow not found"}

        workflow = self.workflows[workflow_id]
        user_id = workflow['user_id']
        dag = workflow['dag']

        # Update workflow status
        workflow['status'] = 'RUNNING'

        # Submit jobs to branch queues
        job_count = 0
        for branch_id, jobs in dag.items():
            for job in jobs:
                job['user_id'] = user_id
                job['workflow_id'] = workflow_id
                job['branch_id'] = branch_id

                self.branch_queues[branch_id].append(job)
                self.user_jobs[user_id].add(job['job_id'])
                job_count += 1

                logger.debug(f"Queued {job['job_id']} in branch {branch_id}")

        logger.info(f"Workflow {workflow_id}: {job_count} jobs queued across {len(dag)} branches")

        return {
            "status": "ACCEPTED",
            "workflow_id": workflow_id,
            "branches": list(dag.keys()),
            "total_jobs": job_count,
            "message": f"Workflow accepted with {job_count} jobs"
        }
    
    
    async def get_next_job(self) -> Optional[dict]:
        """
        Get next executable job following these rules:
        1. Global worker limit not exceeded
        2. Branch not currently busy
        3. FIFO order within branch
        
        Returns:
            Job dict or None if no job available
        """
        # Check global limit
        if len(self.active_jobs) >= self.max_workers:
            return None
        
        # Try each branch
        for branch_id, queue in list(self.branch_queues.items()):
            # Skip if branch is busy (already has running job)
            if branch_id in self.branch_active:
                continue
            
            # Get next job from this branch
            if queue:
                job = queue.popleft()
                job_id = job['job_id']
                
                # Mark branch as busy
                self.branch_active[branch_id] = job_id
                
                # Mark job as active
                self.active_jobs[job_id] = job
                
                logger.info(f"Dispatching job {job_id} from branch {branch_id} "
                           f"(active: {len(self.active_jobs)}/{self.max_workers})")
                
                return job
        
        return None
    
    
    async def job_completed(self, job_id: str, status: str, result: dict = None):
        """
        Mark job as completed, free branch for next job

        Args:
            job_id: Job identifier
            status: 'SUCCESS' or 'FAILED'
            result: Optional result data from the job execution
        """
        if job_id not in self.active_jobs:
            logger.warning(f"Job {job_id} not found in active jobs")
            return

        job = self.active_jobs[job_id]
        branch_id = job['branch_id']
        user_id = job['user_id']
        workflow_id = job['workflow_id']

        logger.info(f"Job {job_id} completed with status {status}")

        # Store result in workflow DAG
        if result and workflow_id in self.workflows:
            workflow = self.workflows[workflow_id]
            # Find and update the job in the DAG
            for branch_jobs in workflow['dag'].values():
                for dag_job in branch_jobs:
                    if dag_job['job_id'] == job_id:
                        dag_job['result'] = result
                        logger.debug(f"Stored result for job {job_id}")
                        break
        
        # Remove from active jobs
        del self.active_jobs[job_id]
        
        # Free branch
        if self.branch_active.get(branch_id) == job_id:
            del self.branch_active[branch_id]
            logger.debug(f"Branch {branch_id} freed")
        
        # Remove from user's jobs
        if user_id in self.user_jobs:
            self.user_jobs[user_id].discard(job_id)
        
        # Check if user has no more jobs
        if not self.has_pending_jobs(user_id):
            logger.info(f"User {user_id} has no more jobs")
            self.active_users.discard(user_id)

            # Update workflow status
            if workflow_id in self.workflows:
                self.workflows[workflow_id]['status'] = 'COMPLETED'

            # Let next user in queue take slot and start their queued workflows
            if self.user_queue:
                next_user = self.user_queue.popleft()
                self.active_users.add(next_user)
                logger.info(f"User {next_user} promoted from queue "
                           f"({len(self.active_users)}/{self.max_active_users} active)")

                # Find and enqueue all QUEUED workflows for this user
                for wf_id, wf in self.workflows.items():
                    if wf.get('user_id') == next_user and wf.get('status') == 'QUEUED':
                        logger.info(f"Re-enqueueing workflow {wf_id} for promoted user {next_user}")
                        await self._enqueue_workflow_jobs(wf_id)
    
    
    def has_pending_jobs(self, user_id: str) -> bool:
        """Check if user has any pending or active jobs"""
        # Check active jobs
        for job_id, job in self.active_jobs.items():
            if job.get('user_id') == user_id:
                return True
        
        # Check queued jobs
        for queue in self.branch_queues.values():
            for job in queue:
                if job.get('user_id') == user_id:
                    return True
        
        return False
    
    
    def get_workflow_status(self, workflow_id: str) -> Optional[dict]:
        """Get workflow status with job progress"""
        if workflow_id not in self.workflows:
            return None

        workflow = self.workflows[workflow_id]

        # Count jobs by status and build branch details
        total_jobs = workflow['total_jobs']
        completed = 0
        running = 0
        queued = 0

        # Build branch information with job statuses
        branches_info = []
        for branch_id, jobs in workflow['dag'].items():
            branch_jobs = []
            branch_completed = 0
            branch_running = 0
            branch_queued = 0

            for job in jobs:
                job_id = job['job_id']
                job_status = 'QUEUED'  # Default

                if job_id in self.active_jobs:
                    job_status = 'RUNNING'
                    running += 1
                    branch_running += 1
                else:
                    # Check if in queue
                    found_in_queue = False
                    for q in self.branch_queues.values():
                        if any(j['job_id'] == job_id for j in q):
                            job_status = 'QUEUED'
                            queued += 1
                            branch_queued += 1
                            found_in_queue = True
                            break

                    if not found_in_queue:
                        job_status = 'COMPLETED'
                        completed += 1
                        branch_completed += 1

                # Add job with status
                branch_jobs.append({
                    'job_id': job['job_id'],
                    'image': job.get('image', ''),
                    'type': job.get('type', 'segment'),
                    'status': job_status
                })

            # Determine branch status
            if branch_completed == len(jobs):
                branch_status = 'COMPLETED'
            elif branch_running > 0:
                branch_status = 'RUNNING'
            else:
                branch_status = 'QUEUED'

            branches_info.append({
                'branch_id': branch_id,
                'jobs': branch_jobs,
                'status': branch_status,
                'active_job': self.branch_active.get(branch_id)
            })

        progress = int((completed / total_jobs) * 100) if total_jobs > 0 else 0

        return {
            "workflow_id": workflow['workflow_id'],
            "user_id": workflow['user_id'],
            "name": workflow['name'],
            "status": workflow['status'],
            "created_at": workflow['created_at'],
            "progress": progress,
            "branches": branches_info,
            "total_jobs": total_jobs,
            "completed_jobs": completed,
            "running_jobs": running,
            "queued_jobs": queued
        }
    
    
    def get_scheduler_stats(self) -> dict:
        """Get overall scheduler statistics"""
        return {
            "active_workers": len(self.active_jobs),
            "max_workers": self.max_workers,
            "active_users": len(self.active_users),
            "max_active_users": self.max_active_users,
            "queued_users": len(self.user_queue),
            "branches_active": len(self.branch_active),
            "total_branches": len(self.branch_queues),
            "workflows_active": len([w for w in self.workflows.values() if w['status'] == 'RUNNING'])
        }


# Global scheduler instance
scheduler = BranchAwareScheduler(max_workers=10, max_active_users=3)
