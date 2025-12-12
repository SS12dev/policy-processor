"""
Simple A2A Agent Server Implementation
This example demonstrates how to create a basic A2A agent server that:
- Handles non-streaming requests
- Processes text messages
- Returns responses asynchronously
"""

import asyncio
from typing import AsyncIterator
from a2a import types
from a2a.server import (
    AgentExecutor,
    RequestContext,
    A2AFastAPIApplication
)


class SimpleAgent(AgentExecutor):
    """
    A simple agent that echoes back messages with additional processing.
    This demonstrates the basic structure of an A2A agent.
    """

    async def execute(
        self,
        request_context: RequestContext
    ) -> AsyncIterator[types.TaskStatusUpdateEvent | types.TaskArtifactUpdateEvent | types.Message]:
        """
        Process incoming messages and yield responses.

        Args:
            request_context: Contains the user message and configuration

        Yields:
            Status updates, artifact updates, or response messages
        """
        # Extract the user's message
        user_message = request_context.user_message

        # Extract text from message parts
        user_text = ""
        for part in user_message.parts:
            if isinstance(part.root, types.TextPart):
                user_text += part.root.text

        print(f"Received message: {user_text}")

        # Yield a status update indicating processing has started
        yield types.TaskStatusUpdateEvent(
            task_id=request_context.task_id,
            context_id=request_context.context_id,
            status=types.TaskStatus(
                state=types.TaskState.WORKING,
                message="Processing your request..."
            ),
            final=False
        )

        # Simulate some processing
        await asyncio.sleep(0.5)

        # Create a response message
        response_text = f"Echo: {user_text}\n\nYou sent a message with {len(user_text)} characters."

        response_message = types.Message(
            message_id=f"msg_{asyncio.get_event_loop().time()}",
            role=types.Role.AGENT,
            parts=[
                types.Part(root=types.TextPart(text=response_text))
            ]
        )

        # Yield the response message
        yield response_message

        # Yield final status update
        yield types.TaskStatusUpdateEvent(
            task_id=request_context.task_id,
            context_id=request_context.context_id,
            status=types.TaskStatus(
                state=types.TaskState.COMPLETED,
                message="Request processed successfully"
            ),
            final=True
        )


def create_agent_server():
    """
    Create and configure the A2A FastAPI server.

    Returns:
        A2AFastAPIApplication instance
    """
    # Define the agent card (agent metadata)
    agent_card = types.AgentCard(
        name="Simple Echo Agent",
        description="A simple agent that echoes back your messages",
        url="http://localhost:8000",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=types.AgentCapabilities(
            streaming=True,  # Supports streaming responses
            push_notifications=False  # Does not support push notifications
        ),
        skills=[
            types.AgentSkill(
                name="echo",
                description="Echo back the input message",
                input_modes=["text"],
                output_modes=["text"]
            )
        ]
    )

    # Create the agent executor
    agent = SimpleAgent()

    # Create the FastAPI application
    # Note: You can pass additional FastAPI configuration here
    app = A2AFastAPIApplication(
        agent_card=agent_card,
        agent_executor=agent,
        # Optional: Configure max message size, timeouts, etc.
        # These are passed to the underlying FastAPI app
    )

    return app


if __name__ == "__main__":
    import uvicorn

    # Create the agent server
    app = create_agent_server()

    print("="*80)
    print("Starting Simple A2A Agent Server")
    print("="*80)
    print("Server will be available at: http://localhost:8000")
    print("Agent card at: http://localhost:8000/.well-known/agent-card.json")
    print("RPC endpoint at: http://localhost:8000/rpc")
    print("="*80)

    # Run the server
    # Note: In production, you would typically run this with:
    # uvicorn 01_simple_agent_server:app --host 0.0.0.0 --port 8000
    uvicorn.run(
        app.app,  # The underlying FastAPI app
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
