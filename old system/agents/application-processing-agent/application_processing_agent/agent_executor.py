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
    from .agent import ApplicationProcessingAgent
except ImportError:
    from agent import ApplicationProcessingAgent

logger = logging.getLogger(__name__)

class ApplicationProcessingAgentExecutor(AgentExecutor):
    """
    A2A protocol bridge for the Application Processing Agent.
    Handles request validation, task lifecycle, and response streaming.
    """

    def __init__(self):
        """Initialize the agent executor."""
        logger.info("Initializing ApplicationProcessingAgentExecutor")
        self.agent = ApplicationProcessingAgent()
        logger.info("ApplicationProcessingAgentExecutor initialized successfully")

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        Execute agent task based on incoming request.

        Args:
            context: Request context with user input
            event_queue: Queue for sending updates to client
        """
        error = self._validate_request(context)
        if error:
            logger.warning("Request validation failed")
            raise ServerError(error=InvalidParamsError(message="Invalid request parameters"))

        user_input = context.get_user_input()
        if not user_input:
            logger.error("No user input found")
            raise ServerError(error=InvalidParamsError(message="No user input found"))

        task = context.current_task
        if not task:
            logger.info("Creating new task")
            task = new_task(context.message)
            await event_queue.enqueue_event(task)
        else:
            logger.info(f"Using existing task: {task.id}")

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            logger.info(f"Starting agent execution for task {task.id}")

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    "Processing application document...",
                    task.context_id,
                    task.id,
                ),
            )

            try:
                request_data = json.loads(user_input)
                task_type = request_data.get("task")
                input_data = request_data.get("data", {})
            except json.JSONDecodeError:
                raise ServerError(error=InvalidParamsError(message="Invalid JSON input"))

            logger.info(f"Task type: {task_type}")

            result = await self.agent.invoke(task_type, input_data)

            if result["status"] == "completed":
                logger.info("Task completed successfully")

                # MEMORY OPTIMIZATION: Use compact JSON without indentation
                result_text = json.dumps(result, separators=(',', ':'))

                await updater.add_artifact(
                    [Part(root=TextPart(text=result_text))],
                    name='application_processing_result',
                )
                await updater.complete()
                
                # MEMORY CLEANUP: Clear large variables immediately
                del result_text
                del result
                del input_data
                gc.collect()

            elif result["status"] == "error":
                logger.error(f"Agent error: {result['message']}")
                await updater.update_status(
                    TaskState.failed,
                    new_agent_text_message(
                        result['message'],
                        task.context_id,
                        task.id,
                    ),
                    final=True
                )
                # MEMORY CLEANUP: Clear variables on error
                del result
                del input_data
                gc.collect()

        except Exception as e:
            logger.error(f'Error during agent execution: {e}', exc_info=True)

            try:
                await updater.update_status(
                    TaskState.failed,
                    new_agent_text_message(
                        "Internal error during application processing. Please try again.",
                        task.context_id,
                        task.id,
                    ),
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
        """
        Validate incoming request.

        Args:
            context: Request context to validate

        Returns:
            True if invalid, False if valid
        """
        user_input = context.get_user_input()
        if not user_input or not user_input.strip():
            logger.warning("No input provided")
            return True

        if len(user_input.strip()) > 10000000:
            logger.warning("Input too long")
            return True

        try:
            request_data = json.loads(user_input)
            if "task" not in request_data:
                logger.warning("No task specified in request")
                return True
        except json.JSONDecodeError:
            logger.warning("Invalid JSON input")
            return True

        return False

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Handle task cancellation."""
        logger.warning("Task cancellation not supported")
        raise ServerError(error=UnsupportedOperationError(message="Task cancellation is not supported"))