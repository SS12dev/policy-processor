"""
Streaming A2A Agent Server Implementation
This example demonstrates:
- Real-time streaming responses
- Progressive content generation
- Status updates during processing
- Artifact streaming (for large outputs)
"""

import asyncio
from typing import AsyncIterator
from a2a import types
from a2a.server import (
    AgentExecutor,
    RequestContext,
    A2AFastAPIApplication
)


class StreamingAgent(AgentExecutor):
    """
    An agent that streams responses word-by-word to demonstrate streaming capabilities.
    This is useful for long-running operations where you want to show progress to the user.
    """

    async def execute(
        self,
        request_context: RequestContext
    ) -> AsyncIterator[types.TaskStatusUpdateEvent | types.TaskArtifactUpdateEvent | types.Message]:
        """
        Process incoming messages and stream responses progressively.

        Args:
            request_context: Contains the user message and configuration

        Yields:
            Status updates, messages, and artifacts as they're generated
        """
        # Extract the user's message
        user_message = request_context.user_message
        user_text = ""

        for part in user_message.parts:
            if isinstance(part.root, types.TextPart):
                user_text += part.root.text

        print(f"[Streaming Agent] Received: {user_text}")

        # Yield initial status
        yield types.TaskStatusUpdateEvent(
            task_id=request_context.task_id,
            context_id=request_context.context_id,
            status=types.TaskStatus(
                state=types.TaskState.WORKING,
                message="Starting to generate response..."
            ),
            final=False
        )

        # Simulate progressive response generation
        response_parts = [
            "I'm processing your message.",
            "Let me think about this carefully.",
            f"You mentioned: '{user_text}'",
            "Here's my detailed response:",
            f"Your message has {len(user_text)} characters and {len(user_text.split())} words.",
            "I hope this information is helpful!"
        ]

        accumulated_text = ""

        for i, part_text in enumerate(response_parts):
            # Simulate processing time
            await asyncio.sleep(0.3)

            # Update status to show progress
            yield types.TaskStatusUpdateEvent(
                task_id=request_context.task_id,
                context_id=request_context.context_id,
                status=types.TaskStatus(
                    state=types.TaskState.WORKING,
                    message=f"Generating response part {i+1}/{len(response_parts)}..."
                ),
                final=False
            )

            # Add to accumulated text
            accumulated_text += part_text + " "

            # Yield a message with the accumulated content
            # This shows progressive updates to the client
            yield types.Message(
                message_id=f"msg_{request_context.task_id}_{i}",
                role=types.Role.AGENT,
                parts=[
                    types.Part(root=types.TextPart(text=accumulated_text.strip()))
                ]
            )

        # Optionally create an artifact with the full response
        # Artifacts are useful for structured outputs like files, reports, etc.
        yield types.TaskArtifactUpdateEvent(
            task_id=request_context.task_id,
            context_id=request_context.context_id,
            artifact=types.Artifact(
                artifact_id=f"artifact_{request_context.task_id}",
                name="response_summary.txt",
                description="Summary of the conversation",
                parts=[
                    types.Part(root=types.TextPart(
                        text=f"User Input: {user_text}\n\nAgent Response: {accumulated_text}"
                    ))
                ]
            ),
            last_chunk=True,  # Indicates this is the final chunk of the artifact
            append=False  # False means replace, True would append to existing artifact
        )

        # Yield final status
        yield types.TaskStatusUpdateEvent(
            task_id=request_context.task_id,
            context_id=request_context.context_id,
            status=types.TaskStatus(
                state=types.TaskState.COMPLETED,
                message="Response generation complete"
            ),
            final=True
        )


def create_streaming_server():
    """
    Create and configure a streaming-capable A2A server.

    Returns:
        A2AFastAPIApplication instance
    """
    agent_card = types.AgentCard(
        name="Streaming Agent",
        description="An agent that demonstrates progressive response streaming",
        url="http://localhost:8001",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text", "artifact"],
        capabilities=types.AgentCapabilities(
            streaming=True,  # CRITICAL: Must be True for streaming support
            push_notifications=False
        ),
        skills=[
            types.AgentSkill(
                name="stream_response",
                description="Generate responses progressively with status updates",
                input_modes=["text"],
                output_modes=["text", "artifact"]
            )
        ]
    )

    agent = StreamingAgent()

    app = A2AFastAPIApplication(
        agent_card=agent_card,
        agent_executor=agent
    )

    return app


if __name__ == "__main__":
    import uvicorn

    app = create_streaming_server()

    print("="*80)
    print("Starting Streaming A2A Agent Server")
    print("="*80)
    print("Server: http://localhost:8001")
    print("Agent card: http://localhost:8001/.well-known/agent-card.json")
    print("RPC endpoint: http://localhost:8001/rpc")
    print("Streaming endpoint: http://localhost:8001/rpc (with message/stream)")
    print("="*80)
    print("\nThis server supports both:")
    print("  - Non-streaming requests (message/send)")
    print("  - Streaming requests (message/stream)")
    print("="*80)

    uvicorn.run(
        app.app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
