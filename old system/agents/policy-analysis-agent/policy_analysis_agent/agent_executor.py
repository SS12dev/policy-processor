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
    from .agent import PolicyAnalysisAgent
except ImportError:
    from agent import PolicyAnalysisAgent

logger = logging.getLogger(__name__)

class PolicyAnalysisAgentExecutor(AgentExecutor):
    """
    A2A protocol bridge for the Policy Analysis Agent.
    Handles request validation, task lifecycle, and response streaming.
    """

    def __init__(self):
        """Initialize the agent executor."""
        logger.info("Initializing PolicyAnalysisAgentExecutor")
        self.agent = PolicyAnalysisAgent()
        logger.info("PolicyAnalysisAgentExecutor initialized successfully")

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
                    "Processing policy analysis request...",
                    task.context_id,
                    task.id,
                ),
            )

            try:
                logger.info(f"Processing user input: {len(user_input)} characters")
                request_data = json.loads(user_input)
                task_type = request_data.get("task")
                input_data = request_data.get("data", {})
                
                # Log task details without sensitive content
                data_summary = {}
                if input_data:
                    for key, value in input_data.items():
                        if isinstance(value, str) and len(value) > 100:
                            data_summary[key] = f"<string:{len(value)} chars>"
                        elif isinstance(value, (list, dict)):
                            data_summary[key] = f"<{type(value).__name__}:{len(value)} items>"
                        else:
                            data_summary[key] = str(value)[:50]
                
                logger.info(f"Task: {task_type}, Input data: {data_summary}")
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON input: {e}")
                raise ServerError(error=InvalidParamsError(message="Invalid JSON format in request"))

            logger.info(f"Task type: {task_type}")

            result = await self.agent.invoke(task_type, input_data)

            if result["status"] == "completed":
                logger.info("Task completed successfully")

                # MEMORY OPTIMIZATION: Use compact JSON without indentation to reduce memory usage
                result_text = json.dumps(result, separators=(',', ':'))

                await updater.add_artifact(
                    [Part(root=TextPart(text=result_text))],
                    name='policy_analysis_result',
                )
                await updater.complete()
                
                # MEMORY CLEANUP: Clear large variables and force garbage collection
                del result_text
                del result
                del input_data  # Clear the original policy data
                gc.collect()  # Force garbage collection to free memory immediately

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
                # MEMORY CLEANUP: Clear variables even on error
                del result
                del input_data
                gc.collect()

        except Exception as e:
            logger.error(f'Error during agent execution: {e}', exc_info=True)

            try:
                await updater.update_status(
                    TaskState.failed,
                    new_agent_text_message(
                        "Internal error during policy analysis. Please try again.",
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
        try:
            user_input = context.get_user_input()
            if not user_input or not user_input.strip():
                logger.warning("Request validation failed: No input provided")
                return True

            # Check input size limits (2MB max for large policy documents)
            input_size = len(user_input.strip())
            max_size = 2000000
            if input_size > max_size:
                logger.warning(f"Request validation failed: Input too large ({input_size} > {max_size} chars)")
                return True

            # Validate JSON structure
            try:
                request_data = json.loads(user_input)
            except json.JSONDecodeError as e:
                logger.warning(f"Request validation failed: Invalid JSON - {e}")
                return True

            # Validate required fields
            if not isinstance(request_data, dict):
                logger.warning("Request validation failed: Root must be JSON object")
                return True

            task_type = request_data.get("task")
            if not task_type:
                logger.warning("Request validation failed: Missing 'task' field")
                return True

            if not isinstance(task_type, str):
                logger.warning("Request validation failed: 'task' must be string")
                return True

            # Validate supported task types
            supported_tasks = ["analyze_policy", "consolidate_questionnaire"]
            if task_type not in supported_tasks:
                logger.warning(f"Request validation failed: Unsupported task '{task_type}'. Supported: {supported_tasks}")
                return True

            # Validate task-specific requirements
            data = request_data.get("data", {})
            if not isinstance(data, dict):
                logger.warning("Request validation failed: 'data' field must be object")
                return True

            # Task-specific validation
            if task_type == "analyze_policy":
                if not data.get("policy_text"):
                    logger.warning("Request validation failed: 'policy_text' required for analyze_policy task")
                    return True

            logger.debug(f"Request validation passed for task: {task_type}")
            return False

        except Exception as e:
            logger.error(f"Unexpected error during request validation: {e}", exc_info=True)
            return True

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Handle task cancellation."""
        logger.warning("Task cancellation not supported")
        raise ServerError(error=UnsupportedOperationError(message="Task cancellation is not supported"))