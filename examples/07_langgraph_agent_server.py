"""
LangGraph A2A Agent Implementation
This example demonstrates:
- Integrating LangGraph with A2A protocol
- Using LangGraph's ReAct agent pattern
- Managing conversation state and memory
- Multi-turn conversations with context
- Streaming LangGraph responses through A2A
"""

import asyncio
from typing import AsyncIterator, Literal
from uuid import uuid4
import re
from pydantic import BaseModel

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool

from a2a import types
from a2a.server import (
    AgentExecutor,
    RequestContext,
    A2AFastAPIApplication
)


# Define tools that the LangGraph agent can use
@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression like "2 + 2" or "10 * 5"

    Returns:
        The result of the calculation
    """
    try:
        # Safe evaluation - only allows basic math operations
        # Remove any non-math characters for safety
        safe_expr = re.sub(r'[^0-9+\-*/().\s]', '', expression)
        result = eval(safe_expr, {"__builtins__": {}}, {})
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


@tool
def get_time_info(timezone: str = "UTC") -> str:
    """
    Get the current time information.

    Args:
        timezone: The timezone to get the time for (default: UTC)

    Returns:
        Current time information
    """
    from datetime import datetime
    now = datetime.now()
    return f"Current time in {timezone}: {now.strftime('%Y-%m-%d %H:%M:%S')}"


# Response format for the agent's structured output
class AgentResponseFormat(BaseModel):
    """Structured format for agent responses."""
    status: Literal["input_required", "completed", "error"]
    message: str


class LangGraphA2AAgent(AgentExecutor):
    """
    An A2A agent powered by LangGraph.
    This demonstrates how to wrap a LangGraph agent as an A2A-compliant agent.
    """

    def __init__(self):
        """Initialize the LangGraph agent with tools and memory."""
        super().__init__()

        # Define available tools
        self.tools = [calculate, get_time_info]

        # System instruction for the agent
        self.system_instruction = """
You are a helpful AI assistant with access to tools.

When the user asks you to perform calculations or get time information,
use the appropriate tools.

Always provide clear, concise responses.

If you need more information from the user, ask clarifying questions.
"""

        # Create memory saver for conversation history
        self.memory = MemorySaver()

        # Create the ReAct agent using LangGraph
        # Note: In production, you would use an actual LLM here
        # For this example, we'll create a simple graph
        self.graph = self._create_agent_graph()

        print("[LangGraph Agent] Initialized with tools:", [tool.name for tool in self.tools])

    def _create_agent_graph(self):
        """
        Create a LangGraph agent graph.
        In production, this would use create_react_agent with a real LLM.
        """
        # For demonstration, we create a simple graph
        # In production, you would do:
        # return create_react_agent(
        #     model=your_llm,  # e.g., ChatOpenAI() or ChatAnthropic()
        #     tools=self.tools,
        #     checkpointer=self.memory,
        #     prompt=self.system_instruction
        # )

        # Simple demo graph that echoes and uses tools if needed
        workflow = StateGraph(MessagesState)

        def agent_node(state: MessagesState) -> MessagesState:
            """Simple agent logic for demonstration."""
            messages = state["messages"]
            last_message = messages[-1]

            if isinstance(last_message, HumanMessage):
                user_text = last_message.content.lower()

                # Simple pattern matching for tool usage
                if "calculate" in user_text or any(op in user_text for op in ['+', '-', '*', '/', '=']):
                    # Extract the calculation
                    import re
                    numbers = re.findall(r'\d+[\+\-\*/]\d+', user_text)
                    if numbers:
                        result = calculate.invoke({"expression": numbers[0]})
                        response = AIMessage(content=result)
                    else:
                        response = AIMessage(
                            content="I can help with calculations. Please provide a math expression like '2 + 2'."
                        )
                elif "time" in user_text:
                    result = get_time_info.invoke({"timezone": "UTC"})
                    response = AIMessage(content=result)
                else:
                    # Echo back with processing
                    response = AIMessage(
                        content=f"I received your message: '{last_message.content}'. "
                                f"I can help you with calculations and time information. "
                                f"What would you like to know?"
                    )

                return {"messages": [response]}

            return {"messages": []}

        workflow.add_node("agent", agent_node)
        workflow.add_edge(START, "agent")
        workflow.add_edge("agent", END)

        return workflow.compile(checkpointer=self.memory)

    def _extract_text_from_message(self, message: types.Message) -> str:
        """Extract text content from an A2A message."""
        text_parts = []
        for part in message.parts:
            if isinstance(part.root, types.TextPart):
                text_parts.append(part.root.text)
        return " ".join(text_parts)

    def _invoke_graph(self, user_text: str, context_id: str) -> dict:
        """
        Invoke the LangGraph agent.

        Args:
            user_text: The user's input text
            context_id: The conversation context ID

        Returns:
            Agent response with status and message
        """
        # Configure the graph run with the context ID as thread_id
        # This ensures conversation history is maintained
        config: RunnableConfig = {
            "configurable": {"thread_id": context_id}
        }

        # Create the input
        inputs = {"messages": [HumanMessage(content=user_text)]}

        # Invoke the graph
        result = self.graph.invoke(inputs, config)

        # Extract the response
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, AIMessage):
                return {
                    "status": "completed",
                    "message": last_message.content
                }

        return {
            "status": "error",
            "message": "No response generated"
        }

    async def execute(
        self,
        request_context: RequestContext
    ) -> AsyncIterator[types.TaskStatusUpdateEvent | types.TaskArtifactUpdateEvent | types.Message]:
        """
        Execute the LangGraph agent for an A2A request.

        Args:
            request_context: The A2A request context

        Yields:
            A2A protocol events (status updates, messages, artifacts)
        """
        # Extract user message
        user_message = request_context.user_message
        user_text = self._extract_text_from_message(user_message)

        print(f"[LangGraph Agent] Processing: {user_text}")
        print(f"[LangGraph Agent] Context ID: {request_context.context_id}")

        # Yield initial status
        yield types.TaskStatusUpdateEvent(
            task_id=request_context.task_id,
            context_id=request_context.context_id,
            status=types.TaskStatus(
                state=types.TaskState.WORKING,
                message="Processing your request with LangGraph agent..."
            ),
            final=False
        )

        # Simulate some processing time
        await asyncio.sleep(0.3)

        # Invoke the LangGraph agent
        try:
            agent_response = await asyncio.to_thread(
                self._invoke_graph,
                user_text,
                request_context.context_id
            )

            response_status = agent_response["status"]
            response_message = agent_response["message"]

            # Create A2A message
            a2a_message = types.Message(
                message_id=f"msg_{uuid4()}",
                role=types.Role.AGENT,
                parts=[
                    types.Part(root=types.TextPart(text=response_message))
                ]
            )

            # Yield the message
            yield a2a_message

            # Determine final task state based on agent response
            if response_status == "completed":
                final_state = types.TaskState.COMPLETED
                final_message = "Request processed successfully"
            elif response_status == "input_required":
                final_state = types.TaskState.INPUT_REQUIRED
                final_message = "Additional input required"
            else:
                final_state = types.TaskState.FAILED
                final_message = "Error processing request"

            # Yield final status
            yield types.TaskStatusUpdateEvent(
                task_id=request_context.task_id,
                context_id=request_context.context_id,
                status=types.TaskStatus(
                    state=final_state,
                    message=final_message
                ),
                final=True
            )

        except Exception as e:
            print(f"[LangGraph Agent] Error: {str(e)}")

            # Yield error status
            yield types.TaskStatusUpdateEvent(
                task_id=request_context.task_id,
                context_id=request_context.context_id,
                status=types.TaskStatus(
                    state=types.TaskState.FAILED,
                    message=f"Error: {str(e)}"
                ),
                final=True
            )


def create_langgraph_a2a_server():
    """
    Create an A2A server with a LangGraph agent.

    Returns:
        A2AFastAPIApplication instance
    """
    agent_card = types.AgentCard(
        name="LangGraph Assistant",
        description="An AI assistant powered by LangGraph with tool-using capabilities",
        url="http://localhost:8003",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=types.AgentCapabilities(
            streaming=True,
            push_notifications=False
        ),
        skills=[
            types.AgentSkill(
                name="calculation",
                description="Perform mathematical calculations",
                input_modes=["text"],
                output_modes=["text"]
            ),
            types.AgentSkill(
                name="time_info",
                description="Get current time information",
                input_modes=["text"],
                output_modes=["text"]
            ),
            types.AgentSkill(
                name="conversation",
                description="Have multi-turn conversations with context awareness",
                input_modes=["text"],
                output_modes=["text"]
            )
        ]
    )

    agent = LangGraphA2AAgent()

    app = A2AFastAPIApplication(
        agent_card=agent_card,
        agent_executor=agent
    )

    return app


if __name__ == "__main__":
    import uvicorn

    app = create_langgraph_a2a_server()

    print("="*80)
    print("Starting LangGraph A2A Agent Server")
    print("="*80)
    print("Server: http://localhost:8003")
    print("Agent card: http://localhost:8003/.well-known/agent-card.json")
    print("RPC endpoint: http://localhost:8003/rpc")
    print("="*80)
    print("\nKey Features:")
    print("  - LangGraph-powered agent with ReAct pattern")
    print("  - Tool usage (calculator, time info)")
    print("  - Multi-turn conversations with memory")
    print("  - Context-aware responses")
    print("="*80)
    print("\nHow LangGraph integrates with A2A:")
    print("  1. A2A handles the protocol layer (messages, tasks, streaming)")
    print("  2. LangGraph handles the agent logic (reasoning, tools, memory)")
    print("  3. Context ID maps to LangGraph thread_id for conversation history")
    print("  4. Agent responses are converted to A2A messages")
    print("="*80)

    uvicorn.run(
        app.app,
        host="0.0.0.0",
        port=8003,
        log_level="info"
    )
