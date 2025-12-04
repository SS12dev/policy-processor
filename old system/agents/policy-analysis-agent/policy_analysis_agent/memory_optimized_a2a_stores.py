"""
Memory-optimized task stores for A2A agents to prevent memory leaks.
Inherits from the official A2A classes but adds automatic cleanup.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from collections import OrderedDict
from a2a.server.context import ServerCallContext
from a2a.server.tasks.task_store import TaskStore
from a2a.server.tasks.push_notification_config_store import PushNotificationConfigStore
from a2a.types import Task, PushNotificationConfig

logger = logging.getLogger(__name__)

class MemoryOptimizedTaskStore(TaskStore):
    """
    Memory-optimized version of InMemoryTaskStore that automatically prunes old tasks.
    Inherits from the official A2A TaskStore interface.
    """

    def __init__(self, max_tasks: int = 10) -> None:
        """Initializes the MemoryOptimizedTaskStore."""
        logger.debug('Initializing MemoryOptimizedTaskStore with max_tasks=%d', max_tasks)
        self.max_tasks = max_tasks
        self.tasks: OrderedDict[str, Task] = OrderedDict()
        self.lock = asyncio.Lock()
        logger.info(f"MemoryOptimizedTaskStore initialized with max_tasks={max_tasks}")

    async def save(
        self, task: Task, context: ServerCallContext | None = None
    ) -> None:
        """Saves or updates a task and prunes old ones if necessary."""
        async with self.lock:
            # Update or add the task
            if task.id in self.tasks:
                # Move to end (most recent)
                self.tasks.move_to_end(task.id)
            self.tasks[task.id] = task
            logger.debug('Task %s saved successfully. Total tasks: %d', task.id, len(self.tasks))
            
            # Prune old tasks if we exceed the limit
            await self._prune_old_tasks()

    async def get(
        self, task_id: str, context: ServerCallContext | None = None
    ) -> Task | None:
        """Retrieves a task from the store by ID."""
        async with self.lock:
            logger.debug('Attempting to get task with id: %s', task_id)
            task = self.tasks.get(task_id)
            if task:
                logger.debug('Task %s retrieved successfully.', task_id)
                # Move to end (mark as recently accessed)
                self.tasks.move_to_end(task_id)
            else:
                logger.debug('Task %s not found in store (may have been pruned).', task_id)
            return task

    async def delete(
        self, task_id: str, context: ServerCallContext | None = None
    ) -> None:
        """Deletes a task from the store by ID."""
        async with self.lock:
            logger.debug('Attempting to delete task with id: %s', task_id)
            if task_id in self.tasks:
                del self.tasks[task_id]
                logger.debug('Task %s deleted successfully.', task_id)
            else:
                logger.warning(
                    'Attempted to delete nonexistent task with id: %s', task_id
                )

    async def _prune_old_tasks(self) -> None:
        """Prune old tasks to keep memory usage under control, but prioritize concurrency."""
        # Only prune if we're significantly over the limit to allow for burst concurrency
        if len(self.tasks) <= self.max_tasks * 1.2:  # Allow 20% over limit before pruning
            return
            
        # Only remove definitively completed tasks
        completed_tasks = []
        for task_id, task in self.tasks.items():
            if hasattr(task, 'state') and task.state:
                state_str = str(task.state).lower()
                # Be more specific about completion states to avoid pruning active tasks
                if any(status in state_str for status in ['completed', 'failed', 'cancelled', 'finished', 'done']):
                    completed_tasks.append(task_id)
        
        # Only remove completed tasks, never force-prune active ones
        tasks_to_remove = completed_tasks[:len(self.tasks) - self.max_tasks]
        for task_id in tasks_to_remove:
            removed_task = self.tasks.pop(task_id, None)
            if removed_task:
                logger.info('Pruned completed task %s (state: %s)', task_id, getattr(removed_task, 'state', 'unknown'))
        
        # Log warning if we still have too many tasks but avoid force-pruning
        if len(self.tasks) > self.max_tasks:
            logger.warning('Task store has %d tasks (limit: %d) but avoiding force-pruning to preserve concurrency', 
                         len(self.tasks), self.max_tasks)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            "total_tasks": len(self.tasks),
            "max_tasks": self.max_tasks,
            "memory_usage_percent": (len(self.tasks) / self.max_tasks) * 100 if self.max_tasks > 0 else 0
        }


class MemoryOptimizedPushNotificationConfigStore(PushNotificationConfigStore):
    """
    Memory-optimized version of InMemoryPushNotificationConfigStore that automatically prunes old configs.
    Inherits from the official A2A PushNotificationConfigStore interface.
    """

    def __init__(self, max_configs: int = 5) -> None:
        """Initializes the MemoryOptimizedPushNotificationConfigStore."""
        self.max_configs = max_configs
        self.lock = asyncio.Lock()
        self._push_notification_infos: OrderedDict[
            str, list[PushNotificationConfig]
        ] = OrderedDict()
        logger.info(f"MemoryOptimizedPushNotificationConfigStore initialized with max_configs={max_configs}")

    async def set_info(
        self, task_id: str, notification_config: PushNotificationConfig
    ) -> None:
        """Sets or updates the push notification configuration for a task."""
        async with self.lock:
            if task_id not in self._push_notification_infos:
                self._push_notification_infos[task_id] = []

            if notification_config.id is None:
                notification_config.id = task_id

            # Remove existing config with same ID
            configs = self._push_notification_infos[task_id]
            for config in configs:
                if config.id == notification_config.id:
                    configs.remove(config)
                    break

            configs.append(notification_config)
            
            # Move to end (most recent)
            self._push_notification_infos.move_to_end(task_id)
            
            logger.debug('Push config set for task %s. Total configs: %d', 
                        task_id, len(self._push_notification_infos))
            
            # Prune old configs if necessary
            await self._prune_old_configs()

    async def get_info(self, task_id: str) -> list[PushNotificationConfig]:
        """Retrieves the push notification configuration for a task."""
        async with self.lock:
            configs = self._push_notification_infos.get(task_id) or []
            if configs:
                # Move to end (mark as recently accessed)
                self._push_notification_infos.move_to_end(task_id)
            return configs

    async def delete_info(
        self, task_id: str, config_id: str | None = None
    ) -> None:
        """Deletes the push notification configuration for a task."""
        async with self.lock:
            if config_id is None:
                config_id = task_id

            if task_id in self._push_notification_infos:
                configurations = self._push_notification_infos[task_id]
                if not configurations:
                    return

                for config in configurations:
                    if config.id == config_id:
                        configurations.remove(config)
                        break

                if len(configurations) == 0:
                    del self._push_notification_infos[task_id]

    async def _prune_old_configs(self) -> None:
        """Prune old configurations to keep memory usage under control, but prioritize concurrency."""
        # Only prune if we're significantly over the limit to allow for burst concurrency
        if len(self._push_notification_infos) <= self.max_configs * 1.2:  # Allow 20% over limit
            return
            
        # Remove oldest configs when necessary, but be conservative
        prune_count = len(self._push_notification_infos) - self.max_configs
        pruned_count = 0
        
        for _ in range(prune_count):
            if len(self._push_notification_infos) > self.max_configs:
                oldest_task_id = next(iter(self._push_notification_infos))
                removed_configs = self._push_notification_infos.pop(oldest_task_id)
                logger.info('Pruned push configs for task %s (had %d configs)', 
                           oldest_task_id, len(removed_configs))
                pruned_count += 1
            else:
                break
                
        logger.debug('Pruned %d push notification config entries to maintain concurrency', pruned_count)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            "total_configs": len(self._push_notification_infos),
            "max_configs": self.max_configs,
            "memory_usage_percent": (len(self._push_notification_infos) / self.max_configs) * 100 if self.max_configs > 0 else 0
        }