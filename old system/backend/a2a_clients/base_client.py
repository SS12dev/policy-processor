import logging
import httpx
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class A2ABaseClient:
    """Base client for communicating with A2A agents"""

    def __init__(self, agent_url: str, timeout: int = 300):
        """
        Initialize A2A client.

        Args:
            agent_url: Base URL of the agent
            timeout: Request timeout in seconds (default 5 minutes for LLM processing)
        """
        self.agent_url = agent_url.rstrip('/')
        self.timeout = timeout
        # Don't create client in __init__ - create fresh for each request

    async def close(self):
        """Close the HTTP client - no-op since we create fresh clients"""
        pass

    async def get_agent_card(self) -> Optional[Dict[str, Any]]:
        """
        Fetch agent card from /.well-known/agent-card.json (A2A standard endpoint)

        Returns:
            Agent card dict or None if failed
        """
        timeout_config = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=5.0)
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            try:
                # Try the standard A2A agent card endpoint first
                url = f"{self.agent_url}/.well-known/agent-card.json"
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                # Fallback to the deprecated endpoint for compatibility
                try:
                    url = f"{self.agent_url}/.well-known/agent.json"
                    response = await client.get(url)
                    response.raise_for_status()
                    return response.json()
                except Exception as e2:
                    logger.error(f"Failed to fetch agent card from {self.agent_url}: {e}, fallback also failed: {e2}")
                    return None

    async def send_request(
        self,
        task: str,
        data: Dict[str, Any],
        context_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send request to A2A agent using simplified approach that works with our agents.

        Args:
            task: Task type
            data: Task data
            context_id: Optional context ID for conversation

        Returns:
            Response dict with status and data
        """
        # Force immediate logging to ensure we see the start
        print(f"BASE_CLIENT_DEBUG: send_request ENTRY for task: {task}")
        logger.error(f"DEBUG: Starting send_request for task: {task}")
        
        try:
            print(f"BASE_CLIENT_DEBUG: Importing A2A modules...")
            # Import A2A modules
            from a2a.client import A2ACardResolver
            from a2a.types import MessageSendParams, SendMessageRequest
            from uuid import uuid4
            import json
            
            try:
                from a2a.client import ClientFactory
            except ImportError:
                ClientFactory = None
                print(f"BASE_CLIENT_DEBUG: ClientFactory not available, will use A2AClient")

            print(f"BASE_CLIENT_DEBUG: Creating HTTP client...")
            # Create a comprehensive timeout configuration
            timeout_config = httpx.Timeout(
                connect=30.0,  # 30 seconds to connect
                read=self.timeout,     # Full timeout for reading response (5 minutes)
                write=30.0,    # 30 seconds to write request
                pool=10.0      # 10 seconds for connection pool
            )
            async with httpx.AsyncClient(timeout=timeout_config) as httpx_client:
                print(f"BASE_CLIENT_DEBUG: Creating resolver for {self.agent_url}")
                # Get agent card to verify agent is available
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.agent_url)
                
                print(f"BASE_CLIENT_DEBUG: Getting agent card...")
                agent_card = await resolver.get_agent_card()
                if not agent_card:
                    print(f"BASE_CLIENT_DEBUG: FAILED to get agent card!")
                    raise Exception(f"Could not resolve agent card for {self.agent_url}")
                
                print(f"BASE_CLIENT_DEBUG: Agent card received: {type(agent_card)}")
                
                # Create A2A client using the working A2AClient approach
                print(f"BASE_CLIENT_DEBUG: Creating A2A client...")
                from a2a.client import A2AClient
                client = A2AClient(agent_card=agent_card, httpx_client=httpx_client)
                print(f"BASE_CLIENT_DEBUG: Using A2AClient")
                
                # Format the message with task and data as JSON (expected by our agent executors)
                print(f"BASE_CLIENT_DEBUG: Formatting message...")
                message_text = json.dumps({"task": task, "data": data})
                print(f"BASE_CLIENT_DEBUG: Message text: {message_text[:100]}...")
                
                # Build proper A2A message
                print(f"BASE_CLIENT_DEBUG: Building A2A message payload...")
                message_payload = {
                    'message': {
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': message_text}],
                        'message_id': uuid4().hex,
                    },
                }
                
                if context_id:
                    message_payload['message']['context_id'] = context_id
                    
                print(f"BASE_CLIENT_DEBUG: Creating SendMessageRequest...")
                request = SendMessageRequest(
                    id=str(uuid4()), 
                    params=MessageSendParams(**message_payload)
                )
                
                print(f"BASE_CLIENT_DEBUG: About to send message...")
                logger.info(f"Sending A2A message to {self.agent_url} for task: {task} (timeout: {self.timeout}s)")
                
                # Send the message and get response
                try:
                    response = await client.send_message(request)
                except httpx.TimeoutException as e:
                    error_msg = f"Timeout after {self.timeout} seconds waiting for response from {self.agent_url}"
                    logger.error(error_msg)
                    print(f"BASE_CLIENT_DEBUG: TIMEOUT ERROR: {error_msg}")
                    raise Exception(f"Timeout Error: {error_msg}")
                except Exception as e:
                    error_msg = f"Communication error with {self.agent_url}: {str(e)}"
                    logger.error(error_msg)
                    print(f"BASE_CLIENT_DEBUG: COMMUNICATION ERROR: {error_msg}")
                    raise Exception(f"Communication Error: {error_msg}")
                print(f"BASE_CLIENT_DEBUG: Response received! Type: {type(response)}")
                logger.info(f"Received A2A response from {self.agent_url}")
                
                print(f"BASE_CLIENT_DEBUG: Starting response parsing...")
                print(f"BASE_CLIENT_DEBUG: Response type: {type(response)}")
                print(f"BASE_CLIENT_DEBUG: Response attributes: {[attr for attr in dir(response) if not attr.startswith('_')]}")
                print(f"BASE_CLIENT_DEBUG: Response has result: {hasattr(response, 'result')}")
                print(f"BASE_CLIENT_DEBUG: Response has task: {hasattr(response, 'task')}")
                
                # Try to get the full response structure using model_dump
                try:
                    if hasattr(response, 'model_dump'):
                        response_dict = response.model_dump()
                        print(f"BASE_CLIENT_DEBUG: Response model_dump keys: {list(response_dict.keys())}")
                        
                        # Check for result in the dumped data
                        if 'result' in response_dict:
                            result_data = response_dict['result']
                            print(f"BASE_CLIENT_DEBUG: Found result in model_dump: {type(result_data)}")
                            if isinstance(result_data, dict) and 'artifacts' in result_data:
                                artifacts = result_data['artifacts']
                                print(f"BASE_CLIENT_DEBUG: Found {len(artifacts)} artifacts in dumped result")
                                for i, artifact in enumerate(artifacts):
                                    if 'parts' in artifact:
                                        parts = artifact['parts']
                                        print(f"BASE_CLIENT_DEBUG: Artifact {i} has {len(parts)} parts")
                                        for j, part in enumerate(parts):
                                            if 'text' in part:
                                                result_text = part['text']
                                                print(f"BASE_CLIENT_DEBUG: Found text in dumped part {j}: {result_text[:100]}...")
                                                try:
                                                    result = json.loads(result_text)
                                                    print(f"BASE_CLIENT_DEBUG: Successfully parsed JSON from model_dump!")
                                                    logger.info(f"Successfully parsed agent response from model_dump")
                                                    return result
                                                except json.JSONDecodeError as e:
                                                    print(f"BASE_CLIENT_DEBUG: JSON decode error in model_dump: {e}")
                                                    continue
                        else:
                            print(f"BASE_CLIENT_DEBUG: No 'result' key in model_dump")
                    else:
                        print(f"BASE_CLIENT_DEBUG: Response doesn't have model_dump method")
                except Exception as e:
                    print(f"BASE_CLIENT_DEBUG: Error accessing model_dump: {e}")
                
                # Fallback: try accessing result attribute directly (even if hasattr returned False)
                try:
                    result_attr = getattr(response, 'result', None)
                    print(f"BASE_CLIENT_DEBUG: Direct result access: {result_attr}")
                    if result_attr:
                        print(f"BASE_CLIENT_DEBUG: Direct result type: {type(result_attr)}")
                        # Try to access artifacts directly from the result object
                        if hasattr(result_attr, 'artifacts') and result_attr.artifacts:
                            print(f"BASE_CLIENT_DEBUG: Found {len(result_attr.artifacts)} artifacts in direct result")
                            for i, artifact in enumerate(result_attr.artifacts):
                                if hasattr(artifact, 'parts') and artifact.parts:
                                    for j, part in enumerate(artifact.parts):
                                        if hasattr(part, 'text') and part.text:
                                            result_text = part.text
                                            print(f"BASE_CLIENT_DEBUG: Found text in direct result part {j}: {result_text[:100]}...")
                                            try:
                                                result = json.loads(result_text)
                                                print(f"BASE_CLIENT_DEBUG: Successfully parsed JSON from direct result access!")
                                                logger.info(f"Successfully parsed agent response")
                                                return result
                                            except json.JSONDecodeError as e:
                                                print(f"BASE_CLIENT_DEBUG: JSON decode error in direct result: {e}")
                                                continue
                except Exception as e:
                    print(f"BASE_CLIENT_DEBUG: Error accessing result directly: {e}")
                
                # Try to extract JSON from response.result.artifacts (correct A2A structure)
                if hasattr(response, 'result') and response.result:
                    print(f"BASE_CLIENT_DEBUG: Found response.result")
                    if hasattr(response.result, 'artifacts') and response.result.artifacts:
                        print(f"BASE_CLIENT_DEBUG: Found {len(response.result.artifacts)} artifacts in response.result")
                        for i, artifact in enumerate(response.result.artifacts):
                            print(f"BASE_CLIENT_DEBUG: Artifact {i} type: {type(artifact)}")
                            if hasattr(artifact, 'parts') and artifact.parts:
                                print(f"BASE_CLIENT_DEBUG: Artifact {i} has {len(artifact.parts)} parts")
                                for j, part in enumerate(artifact.parts):
                                    print(f"BASE_CLIENT_DEBUG: Part {j} type: {type(part)}")
                                    # Check for direct text attribute (correct structure)
                                    if hasattr(part, 'text') and part.text:
                                        result_text = part.text
                                        print(f"BASE_CLIENT_DEBUG: Found text in part {j}: {result_text[:100]}...")
                                        try:
                                            result = json.loads(result_text)
                                            print(f"BASE_CLIENT_DEBUG: Successfully parsed JSON from response.result.artifacts!")
                                            logger.info(f"Successfully parsed agent response")
                                            return result
                                        except json.JSONDecodeError as e:
                                            print(f"BASE_CLIENT_DEBUG: JSON decode error: {e}")
                                            logger.error(f"JSON decode error: {e}")
                                            continue
                                    # Fallback: check for root.text structure
                                    elif hasattr(part, 'root') and part.root and hasattr(part.root, 'text'):
                                        result_text = part.root.text
                                        print(f"BASE_CLIENT_DEBUG: Found text in part {j} root: {result_text[:100]}...")
                                        try:
                                            result = json.loads(result_text)
                                            print(f"BASE_CLIENT_DEBUG: Successfully parsed JSON from root!")
                                            logger.info(f"Successfully parsed agent response")
                                            return result
                                        except json.JSONDecodeError as e:
                                            print(f"BASE_CLIENT_DEBUG: JSON decode error in root: {e}")
                                            logger.error(f"JSON decode error in root: {e}")
                                            continue
                    else:
                        print(f"BASE_CLIENT_DEBUG: No artifacts in response.result")
                        # Check if result has task attribute (alternative structure)
                        if hasattr(response.result, 'task') and response.result.task:
                            print(f"BASE_CLIENT_DEBUG: Found response.result.task (alternative structure)")
                            task = response.result.task
                            if hasattr(task, 'artifacts') and task.artifacts:
                                print(f"BASE_CLIENT_DEBUG: Found {len(task.artifacts)} artifacts in response.result.task")
                                for i, artifact in enumerate(task.artifacts):
                                    if hasattr(artifact, 'parts') and artifact.parts:
                                        for j, part in enumerate(artifact.parts):
                                            if hasattr(part, 'text') and part.text:
                                                result_text = part.text
                                                try:
                                                    result = json.loads(result_text)
                                                    print(f"BASE_CLIENT_DEBUG: Successfully parsed JSON from response.result.task.artifacts!")
                                                    return result
                                                except json.JSONDecodeError:
                                                    continue
                            else:
                                print(f"BASE_CLIENT_DEBUG: No artifacts in response.result.task")
                else:
                    print(f"BASE_CLIENT_DEBUG: No response.result found")
                
                # Try direct task access
                if hasattr(response, 'task') and response.task:
                    print(f"BASE_CLIENT_DEBUG: Checking response.task directly")
                    task = response.task
                    if hasattr(task, 'artifacts') and task.artifacts:
                        print(f"BASE_CLIENT_DEBUG: Found {len(task.artifacts)} artifacts in response.task")
                        for i, artifact in enumerate(task.artifacts):
                            print(f"BASE_CLIENT_DEBUG: Direct task artifact {i} type: {type(artifact)}")
                            if hasattr(artifact, 'parts') and artifact.parts:
                                print(f"BASE_CLIENT_DEBUG: Direct task artifact {i} has {len(artifact.parts)} parts")
                                for j, part in enumerate(artifact.parts):
                                    if hasattr(part, 'root') and part.root and hasattr(part.root, 'text'):
                                        result_text = part.root.text
                                        print(f"BASE_CLIENT_DEBUG: Found text in direct task part {j}: {result_text[:100]}...")
                                        try:
                                            result = json.loads(result_text)
                                            print(f"BASE_CLIENT_DEBUG: Successfully parsed JSON from direct task!")
                                            logger.info(f"Successfully parsed agent response from direct task")
                                            return result
                                        except json.JSONDecodeError as e:
                                            print(f"BASE_CLIENT_DEBUG: JSON decode error in direct task: {e}")
                                            logger.error(f"JSON decode error in direct task: {e}")
                                            continue
                
                # Try to access other possible response locations based on A2A protocol
                # Check if response has any other task-related attributes
                for attr_name in ['task', 'result', 'message', 'data']:
                    if hasattr(response, attr_name):
                        attr_value = getattr(response, attr_name)
                        print(f"BASE_CLIENT_DEBUG: Found response.{attr_name}: {type(attr_value)}")
                        if attr_value and hasattr(attr_value, '__dict__'):
                            print(f"BASE_CLIENT_DEBUG: response.{attr_name} attributes: {[a for a in dir(attr_value) if not a.startswith('_')]}")
                
                # Check if this is a streaming response that we need to wait for completion
                # In A2A protocol, sometimes we need to check for completed tasks differently
                print(f"BASE_CLIENT_DEBUG: Checking if response indicates task completion...")
                
                # Check for any task completion indicators
                if hasattr(response, 'task_id'):
                    print(f"BASE_CLIENT_DEBUG: Response has task_id: {response.task_id}")
                
                # The issue might be that we're getting an immediate response but the task
                # completes asynchronously. Let's check if there's a way to get the final result
                print(f"BASE_CLIENT_DEBUG: Response might be incomplete - checking for async completion patterns...")
                
                # Fallback: Try to get from response.task.artifacts (old location check)
                if hasattr(response, 'task') and response.task and hasattr(response.task, 'artifacts'):
                    logger.debug(f"Found task with {len(response.task.artifacts)} artifacts")
                    for i, artifact in enumerate(response.task.artifacts):
                        logger.debug(f"Artifact {i}: {type(artifact)}, has parts: {hasattr(artifact, 'parts')}")
                        if hasattr(artifact, 'parts'):
                            for j, part in enumerate(artifact.parts):
                                logger.debug(f"Part {j}: {type(part)}, has root: {hasattr(part, 'root')}")
                                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                    result_text = part.root.text
                                    logger.debug(f"Found text in part {j}: {result_text[:100]}...")
                                    try:
                                        result = json.loads(result_text)
                                        logger.info(f"Successfully parsed agent response from task artifacts")
                                        return result
                                    except json.JSONDecodeError as e:
                                        logger.debug(f"JSON decode error in artifact: {e}")
                                        pass
                
                # Fallback: try other response locations
                response_locations = [
                    (lambda r: r.result if hasattr(r, 'result') else None, "result"),
                    (lambda r: r.message if hasattr(r, 'message') else None, "message"),
                ]
                
                logger.debug(f"Trying fallback response locations...")
                
                for get_obj, location in response_locations:
                    obj = get_obj(response)
                    logger.debug(f"Checking {location}: obj={obj}, has_parts={hasattr(obj, 'parts') if obj else False}")
                    if obj and hasattr(obj, 'parts'):
                        for i, part in enumerate(obj.parts):
                            text = None
                            if hasattr(part, 'text'):
                                text = part.text
                                logger.debug(f"Found text in {location} part {i} (direct): {text[:100] if text else 'None'}...")
                            elif hasattr(part, 'root') and hasattr(part.root, 'text'):
                                text = part.root.text
                                logger.debug(f"Found text in {location} part {i} (root): {text[:100] if text else 'None'}...")
                            
                            if text:
                                try:
                                    result = json.loads(text)
                                    logger.info(f"Successfully parsed agent response from {location}")
                                    return result
                                except json.JSONDecodeError as e:
                                    logger.debug(f"JSON decode error in {location}: {e}")
                                    # Try returning as plain text response
                                    return {
                                        "status": "completed",
                                        "message": text,
                                        "data": {}
                                    }
                
                # If nothing worked, return error
                logger.warning(f"Could not extract valid response from agent")
                return {
                    "status": "error",
                    "message": "No valid response from agent",
                    "data": {}
                }

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from {self.agent_url}: {e.response.status_code}")
            return {
                "status": "error",
                "message": f"HTTP error: {e.response.status_code}",
                "data": {}
            }
        except Exception as e:
            logger.error(f"Error communicating with agent at {self.agent_url}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "data": {}
            }

    async def health_check(self) -> bool:
        """
        Check if agent is healthy.
        
        First tries the dedicated /health endpoint, then falls back to agent card check.

        Returns:
            True if agent is reachable and healthy
        """
        try:
            # Try dedicated health endpoint first (more efficient)
            health_url = f"{self.agent_url}/health"
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(health_url)
                if response.status_code == 200:
                    health_data = response.json()
                    return health_data.get("status") == "healthy"
        except Exception:
            # Health endpoint not available, continue to agent card check
            pass
            
        try:
            # Fallback to agent card check
            card = await self.get_agent_card()
            return card is not None
        except Exception as e:
            logger.error(f"Health check failed for {self.agent_url}: {e}")
            return False