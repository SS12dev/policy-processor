"""
Streaming A2A Client Implementation
This example demonstrates:
- Real-time streaming from server
- Processing progressive updates
- Handling intermediate status updates
- Receiving artifacts as they're generated
"""

import asyncio
import httpx
from a2a import types
from a2a.client import Client, ClientConfig, JsonRpcHttpClientTransport


async def simple_streaming_client():
    """
    Example of a streaming client that receives progressive updates.
    """
    print("="*80)
    print("Streaming A2A Client Example")
    print("="*80)

    async with httpx.AsyncClient(timeout=60.0) as http_client:
        # Fetch the agent card
        agent_card_url = "http://localhost:8001/.well-known/agent-card.json"

        print(f"\nFetching agent card from: {agent_card_url}")
        card_response = await http_client.get(agent_card_url)
        agent_card = types.AgentCard(**card_response.json())

        print(f"Connected to agent: {agent_card.name}")
        print(f"Supports streaming: {agent_card.capabilities.streaming}")

        if not agent_card.capabilities.streaming:
            print("\nWARNING: This agent doesn't support streaming!")
            return

        # Create transport
        transport = JsonRpcHttpClientTransport(
            http_client=http_client,
            base_url="http://localhost:8001/rpc"
        )

        # Configure client for streaming
        client_config = ClientConfig(
            prefer_streaming=True,  # Prefer streaming if available
        )

        client = Client.create(
            card=agent_card,
            config=client_config,
            transport=transport
        )

        # Create message
        user_message = types.Message(
            message_id="streaming_msg_001",
            role=types.Role.USER,
            parts=[
                types.Part(root=types.TextPart(
                    text="Generate a detailed response with multiple parts."
                ))
            ]
        )

        print(f"\nSending message: {user_message.parts[0].root.text}")
        print("\n--- Streaming Response ---\n")

        # Track the latest message content
        latest_message_text = ""
        status_updates = []
        artifacts = []

        # Send message and process stream
        async for event in client.send_message(user_message):
            if isinstance(event, tuple):
                task, update = event

                if isinstance(update, types.TaskStatusUpdateEvent):
                    status_updates.append(update)
                    print(f"[Status] {update.status.state.value}: {update.status.message}")
                    if update.final:
                        print("[Status] Task completed!")

                elif isinstance(update, types.TaskArtifactUpdateEvent):
                    artifacts.append(update)
                    print(f"\n[Artifact] {update.artifact.name}")
                    if update.artifact.description:
                        print(f"  Description: {update.artifact.description}")

                    # Print artifact content
                    for part in update.artifact.parts:
                        if isinstance(part.root, types.TextPart):
                            print(f"  Content preview: {part.root.text[:100]}...")

                    if update.last_chunk:
                        print("  [Last chunk of this artifact]")

            elif isinstance(event, types.Message):
                # This is a message update (could be progressive)
                for part in event.parts:
                    if isinstance(part.root, types.TextPart):
                        # In streaming, you might receive progressive updates
                        # Each update might contain more content than the last
                        current_text = part.root.text

                        # Print only the new content
                        if current_text != latest_message_text:
                            new_content = current_text[len(latest_message_text):]
                            print(new_content, end='', flush=True)
                            latest_message_text = current_text

        print("\n\n--- Stream Complete ---")
        print(f"Total status updates: {len(status_updates)}")
        print(f"Total artifacts received: {len(artifacts)}")
        print(f"Final message length: {len(latest_message_text)} characters")


async def streaming_with_consumer():
    """
    Example using event consumers for processing streaming responses.
    Consumers allow you to react to events as they arrive.
    """
    print("\n\n" + "="*80)
    print("Streaming Client with Event Consumer")
    print("="*80)

    # Define a consumer function that processes events
    async def event_consumer(event, card):
        """
        This function is called for every event received from the agent.

        Args:
            event: The event (Task tuple or Message)
            card: The agent card
        """
        if isinstance(event, tuple):
            task, update = event
            if isinstance(update, types.TaskStatusUpdateEvent):
                if update.status.state == types.TaskState.WORKING:
                    print(f"[Consumer] Agent is working: {update.status.message}")
                elif update.status.state == types.TaskState.COMPLETED:
                    print(f"[Consumer] Agent completed the task!")

        elif isinstance(event, types.Message):
            # Process message
            for part in event.parts:
                if isinstance(part.root, types.TextPart):
                    print(f"[Consumer] Received text: {part.root.text[:50]}...")

    async with httpx.AsyncClient(timeout=60.0) as http_client:
        card_response = await http_client.get(
            "http://localhost:8001/.well-known/agent-card.json"
        )
        agent_card = types.AgentCard(**card_response.json())

        transport = JsonRpcHttpClientTransport(
            http_client=http_client,
            base_url="http://localhost:8001/rpc"
        )

        client_config = ClientConfig(prefer_streaming=True)

        # Create client with the consumer
        client = Client.create(
            card=agent_card,
            config=client_config,
            transport=transport,
            consumers=[event_consumer]  # Add the consumer function
        )

        user_message = types.Message(
            message_id="streaming_msg_002",
            role=types.Role.USER,
            parts=[
                types.Part(root=types.TextPart(
                    text="Process this with the consumer pattern."
                ))
            ]
        )

        print("\nSending message with consumer...\n")

        # The consumer will be called automatically for each event
        async for event in client.send_message(user_message):
            # You can still process events here in addition to the consumer
            pass

        print("\nConsumer processing complete!")


async def streaming_with_reconnection():
    """
    Example demonstrating task resubscription (reconnecting to ongoing tasks).
    This is useful if connection is lost and you want to resume receiving updates.
    """
    print("\n\n" + "="*80)
    print("Streaming with Reconnection Support")
    print("="*80)

    async with httpx.AsyncClient(timeout=60.0) as http_client:
        card_response = await http_client.get(
            "http://localhost:8001/.well-known/agent-card.json"
        )
        agent_card = types.AgentCard(**card_response.json())

        transport = JsonRpcHttpClientTransport(
            http_client=http_client,
            base_url="http://localhost:8001/rpc"
        )

        client_config = ClientConfig(prefer_streaming=True)
        client = Client.create(
            card=agent_card,
            config=client_config,
            transport=transport
        )

        user_message = types.Message(
            message_id="streaming_msg_003",
            role=types.Role.USER,
            parts=[
                types.Part(root=types.TextPart(
                    text="Start a task that I might reconnect to."
                ))
            ]
        )

        print("\nStarting task...")

        task_id = None

        # Start receiving updates
        try:
            async for event in client.send_message(user_message):
                if isinstance(event, tuple):
                    task, update = event
                    task_id = task.id
                    print(f"Task ID: {task_id}")

                    # Simulate connection loss after first update
                    if task_id:
                        print("\nSimulating connection loss...\n")
                        break
        except Exception as e:
            print(f"Connection error: {e}")

        if task_id:
            print(f"Reconnecting to task: {task_id}...")

            # Resubscribe to the task to continue receiving updates
            async for event in client.resubscribe(
                types.TaskIdParams(task_id=task_id)
            ):
                task, update = event
                if isinstance(update, types.TaskStatusUpdateEvent):
                    print(f"[Reconnected] Status: {update.status.state}")
                    if update.final:
                        print("[Reconnected] Task completed!")
                        break

        print("\nReconnection example complete!")


if __name__ == "__main__":
    # Note: Make sure the streaming server is running before executing this client
    # You can run: python 02_streaming_agent_server.py

    print("Make sure the streaming A2A server is running on localhost:8001")
    print("You can start it with: python 02_streaming_agent_server.py\n")

    # Run the examples
    asyncio.run(simple_streaming_client())
    asyncio.run(streaming_with_consumer())
    asyncio.run(streaming_with_reconnection())
