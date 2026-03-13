import time
import uuid
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GenerationTask:
    """Represents a generation task with all necessary information."""
    task_id: str
    modality: str   # "image", "video"
    status: TaskStatus
    prompt: str
    parameters: Dict[str, Any]
    webhook_url: str
    signed_by: str
    created_at: float
    
    # Optional fields
    input_data: Optional[bytes] = None
    result_data: Optional[bytes] = None
    result_url: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation."""
        data = asdict(self)
        data['status'] = self.status.value
        
        # Add processing time if available
        if self.started_at and self.completed_at:
            data['processing_time'] = self.completed_at - self.started_at
        elif self.started_at:
            data['processing_time'] = time.time() - self.started_at
        
        return data


class TaskManager:
    """
    Simple task manager for handling generation tasks.
    
    Features:
    - Thread-safe task operations
    - Automatic cleanup of old tasks
    - Status tracking through task lifecycle
    """
    
    def __init__(self, max_task_age_hours: int = 24):
        self.tasks: Dict[str, GenerationTask] = {}
        self.max_task_age_hours = max_task_age_hours
        self._lock = threading.Lock()
    
    def create_task(
        self,
        modality: str,
        prompt: str,
        parameters: Dict[str, Any],
        webhook_url: str,
        signed_by: str,
        input_data: Optional[bytes] = None,
    ) -> str:
        """Create a new task and return its ID."""
        task_id = str(uuid.uuid4())
        
        task = GenerationTask(
            task_id=task_id,
            modality=modality,
            status=TaskStatus.PENDING,
            prompt=prompt,
            parameters=parameters,
            webhook_url=webhook_url,
            signed_by=signed_by,
            created_at=time.time(),
            input_data=input_data,
        )
        
        with self._lock:
            self.tasks[task_id] = task
        
        return task_id
    
    def get_task(self, task_id: str) -> Optional[GenerationTask]:
        """Get a task by ID."""
        with self._lock:
            return self.tasks.get(task_id)
    
    def get_pending_tasks(self) -> List[GenerationTask]:
        """Get all tasks with PENDING status."""
        with self._lock:
            return [
                task for task in self.tasks.values()
                if task.status == TaskStatus.PENDING
            ]
    
    def mark_task_processing(self, task_id: str) -> bool:
        """Mark a task as processing."""
        with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = TaskStatus.PROCESSING
                task.started_at = time.time()
                return True
            return False
    
    def mark_task_completed(
        self, 
        task_id: str, 
        result_data: Optional[bytes] = None,
        result_url: Optional[str] = None
    ) -> bool:
        """Mark a task as completed with result data."""
        with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                task.result_data = result_data
                task.result_url = result_url
                return True
            return False
    
    def mark_task_failed(self, task_id: str, error_message: str) -> bool:
        """Mark a task as failed with an error message."""
        with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = TaskStatus.FAILED
                task.completed_at = time.time()
                task.error_message = error_message
                return True
            return False
    
    def cleanup_old_tasks(self) -> int:
        """Remove tasks older than max_task_age_hours. Returns number of tasks removed."""
        cutoff_time = time.time() - (self.max_task_age_hours * 3600)
        
        with self._lock:
            old_task_ids = [
                task_id for task_id, task in self.tasks.items()
                if task.created_at < cutoff_time
            ]
            
            for task_id in old_task_ids:
                del self.tasks[task_id]
        
        return len(old_task_ids)
    
    def get_task_stats(self) -> Dict[str, int]:
        """Get statistics about current tasks."""
        with self._lock:
            stats = {"total": len(self.tasks)}
            
            for status in TaskStatus:
                stats[status.value] = sum(
                    1 for task in self.tasks.values()
                    if task.status == status
                )
            
            return stats
