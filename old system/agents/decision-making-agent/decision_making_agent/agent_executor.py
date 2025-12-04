import logging
import json
import gc

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import (
    new_agent_text_message,
    new_task,
)
from a2a.utils.errors import ServerError

try:
    from .agent import DecisionMakingAgent
except ImportError:
    from agent import DecisionMakingAgent

logger = logging.getLogger(__name__)

class DecisionMakingAgentExecutor(AgentExecutor):
    """A2A protocol bridge for the Decision Making Agent."""

    def __init__(self):
        """Initialize the agent executor."""
        logger.info("Initializing DecisionMakingAgentExecutor")
        self.agent = DecisionMakingAgent()
        logger.info("DecisionMakingAgentExecutor initialized successfully")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        error = self._validate_request(context)
        if error:
            raise ServerError(error=InvalidParamsError(message="Invalid request parameters"))

        user_input = context.get_user_input()
        if not user_input:
            raise ServerError(error=InvalidParamsError(message="No user input found"))

        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("Analyzing decision criteria...", task.context_id, task.id),
            )

            request_data = json.loads(user_input)
            task_type = request_data.get("task")
            input_data = request_data.get("data", {})

            result = await self.agent.invoke(task_type, input_data)

            if result["status"] == "completed":
                # MEMORY OPTIMIZATION: Use compact JSON without indentation
                result_text = json.dumps(result, separators=(',', ':'))
                await updater.add_artifact(
                    [Part(root=TextPart(text=result_text))],
                    name='decision_result',
                )
                await updater.complete()
                
                # MEMORY CLEANUP: Clear large variables immediately
                del result_text
                del result
                del input_data
                gc.collect()
                
            elif result["status"] == "error":
                await updater.update_status(
                    TaskState.failed,
                    new_agent_text_message(result['message'], task.context_id, task.id),
                    final=True
                )
                # MEMORY CLEANUP: Clear variables on error
                del result
                del input_data
                gc.collect()

        except Exception as e:
            logger.error(f'Error during execution: {e}', exc_info=True)
            try:
                await updater.update_status(
                    TaskState.failed,
                    new_agent_text_message("Internal error during decision analysis.", task.context_id, task.id),
                    final=True
                )
            except Exception as e2:
                logger.error(f'Failed to send error status: {e2}')
            
            # MEMORY CLEANUP: Clear variables on exception
            try:
                if 'input_data' in locals():
                    del input_data
                if 'result' in locals():
                    del result
                gc.collect()
            except:
                pass  # Ignore cleanup errors
                
            raise ServerError(error=InternalError(message="Internal agent execution error")) from e

    def _validate_request(self, context: RequestContext) -> bool:
        user_input = context.get_user_input()
        if not user_input or not user_input.strip():
            return True
        try:
            request_data = json.loads(user_input)
            if "task" not in request_data:
                return True
        except json.JSONDecodeError:
            return True
        return False

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError(message="Task cancellation is not supported"))