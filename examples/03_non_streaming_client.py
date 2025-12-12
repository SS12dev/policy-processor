"""
Non-Streaming A2A Client Implementation
This example demonstrates:
- Creating an A2A client
- Sending non-streaming messages
- Receiving complete responses
- Handling different response types
"""

import asyncio
import httpx
from a2a import types
from a2a.client import Client, ClientConfig, JsonRpcHttpClientTransport


async def simple_non_streaming_client():
    """
    Example of a simple non-streaming client interaction.
    The client sends a message and waits for the complete response.
    """
    print("="*80)
    print("Non-Streaming A2A Client Example")
    print("="*80)

    # Create an HTTP client (reusable across requests)
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        # Fetch the agent card to discover capabilities
        agent_card_url = "http://localhost:8000/.well-known/agent-card.json"

        print(f"\nFetching agent card from: {agent_card_url}")
        card_response = await http_client.get(agent_card_url)
        agent_card_dict = card_response.json()
        agent_card = types.AgentCard(**agent_card_dict)

        print(f"Connected to agent: {agent_card.name}")
        print(f"Description: {agent_card.description}")
        print(f"Supports streaming: {agent_card.capabilities.streaming}")

        # Create the transport layer
        transport = JsonRpcHttpClientTransport(
            http_client=http_client,
            base_url="http://localhost:8000/rpc"
        )

        # Create the client with configuration
        client_config = ClientConfig(
            # prefer_streaming=False ensures non-streaming even if server supports it
            prefer_streaming=False,
            # You can also configure timeouts, retries, etc.
        )

        client = Client.create(
            card=agent_card,
            config=client_config,
            transport=transport
        )

        # Create a message to send
        user_message = types.Message(
            message_id="msg_001",
            role=types.Role.USER,
            parts=[
                types.Part(root=types.TextPart(
                    text="Hello, this is a non-streaming request. Can you process this?"
                ))
            ]
        )

        print(f"\nSending message: {user_message.parts[0].root.text}")
        print("\nWaiting for response...")

        # Send the message - this is NON-STREAMING
        # The send_message method returns an async iterator, but for non-streaming
        # it will yield the final response only once
        response_count = 0
        async for event in client.send_message(user_message):
            response_count += 1

            # Handle different event types
            if isinstance(event, tuple):
                task, update = event

                if isinstance(update, types.TaskStatusUpdateEvent):
                    print(f"\n[Status Update {response_count}]")
                    print(f"  State: {update.status.state}")
                    print(f"  Message: {update.status.message}")
                    print(f"  Final: {update.final}")

                elif isinstance(update, types.TaskArtifactUpdateEvent):
                    print(f"\n[Artifact Update {response_count}]")
                    print(f"  Artifact ID: {update.artifact.artifact_id}")
                    print(f"  Name: {update.artifact.name}")
                    print(f"  Description: {update.artifact.description}")

            elif isinstance(event, types.Message):
                print(f"\n[Message {response_count}]")
                print(f"  Role: {event.role}")
                for i, part in enumerate(event.parts):
                    if isinstance(part.root, types.TextPart):
                        print(f"  Part {i+1} (Text): {part.root.text}")
                    elif isinstance(part.root, types.DataPart):
                        print(f"  Part {i+1} (Data): {part.root.data}")

        print(f"\n\nTotal events received: {response_count}")
        print("\nConnection closed.")


async def non_streaming_with_configuration():
    """
    Example showing various configuration options for non-streaming clients.
    """
    print("\n\n" + "="*80)
    print("Non-Streaming Client with Advanced Configuration")
    print("="*80)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0),
        limits=httpx.Limits(max_connections=100)
    ) as http_client:
        # Fetch agent card
        card_response = await http_client.get(
            "http://localhost:8000/.well-known/agent-card.json"
        )
        agent_card = types.AgentCard(**card_response.json())

        transport = JsonRpcHttpClientTransport(
            http_client=http_client,
            base_url="http://localhost:8000/rpc"
        )

        # Configure client with specific options
        client_config = ClientConfig(
            prefer_streaming=False,
            # Polling configuration (used when streaming is not available)
            poll_interval_seconds=1.0,
            max_poll_attempts=60
        )

        client = Client.create(
            card=agent_card,
            config=client_config,
            transport=transport
        )

        # Create a message with configuration
        user_message = types.Message(
            message_id="msg_002",
            role=types.Role.USER,
            parts=[
                types.Part(root=types.TextPart(
                    text="This message has custom configuration."
                ))
            ],
            # Metadata can be used to pass additional context
            metadata={
                "user_id": "user_123",
                "session_id": "session_456"
            }
        )

        # Send with message-specific configuration
        configuration = types.MessageSendConfiguration(
            blocking=True,  # Wait for complete response before returning
            history_length=10,  # Include last 10 messages in context
            accepted_output_modes=["text", "data"]  # Specify desired output formats
        )

        print(f"\nSending configured message...")

        async for event in client.send_message(
            user_message,
            configuration=configuration
        ):
            if isinstance(event, types.Message):
                print(f"\nReceived response:")
                for part in event.parts:
                    if isinstance(part.root, types.TextPart):
                        print(f"  {part.root.text}")

        print("\nDone!")


if __name__ == "__main__":
    # Note: Make sure the server is running before executing this client
    # You can run: python 01_simple_agent_server.py

    print("Make sure the A2A server is running on localhost:8000")
    print("You can start it with: python 01_simple_agent_server.py\n")

    # Run the examples
    asyncio.run(simple_non_streaming_client())
    asyncio.run(non_streaming_with_configuration())
