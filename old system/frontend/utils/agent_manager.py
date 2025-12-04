"""Agent Management Service for Prior Authorization System."""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple
import httpx

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from backend.a2a_clients.client_factory import ClientFactory
from backend.settings import backend_settings

logger = logging.getLogger(__name__)


class AgentManager:
    """Manages both local and deployed agents for the Prior Authorization system."""
    
    def __init__(self):
        """Initialize the agent manager."""
        self.local_agent_processes = {}
        self.system_root = Path(__file__).parent.parent.parent
        
    async def check_agent_health(self, agent_url: str, timeout: int = 10) -> Tuple[bool, str]:
        """
        Check health of a single agent.
        
        Args:
            agent_url: URL of the agent to check
            timeout: Request timeout in seconds
            
        Returns:
            Tuple of (is_healthy, status_message)
        """
        try:
            timeout_config = httpx.Timeout(connect=5.0, read=timeout, write=5.0, pool=5.0)
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                health_url = f"{agent_url.rstrip('/')}/health"
                response = await client.get(health_url)
                response.raise_for_status()
                
                # Try to parse response
                try:
                    data = response.json()
                    status = data.get("status", "unknown")
                    return True, f"Healthy - {status}"
                except:
                    return True, "Healthy - Response received"
                    
        except httpx.TimeoutException:
            return False, "Timeout - Agent not responding"
        except httpx.ConnectError:
            return False, "Connection failed - Agent offline"
        except httpx.HTTPStatusError as e:
            return False, f"HTTP {e.response.status_code} - Service error"
        except Exception as e:
            return False, f"Error - {str(e)}"
    
    async def check_all_agents_health(self) -> Dict[str, Dict]:
        """
        Check health of all agents based on current mode.
        
        Returns:
            Dict with agent status information
        """
        results = {}
        
        if backend_settings.USE_DEPLOYED_AGENTS:
            # Check deployed agents
            agents = {
                "Policy Analysis": backend_settings.DEPLOYED_POLICY_AGENT_URL,
                "Application Processing": backend_settings.DEPLOYED_APPLICATION_AGENT_URL,
                "Decision Making": backend_settings.DEPLOYED_DECISION_AGENT_URL
            }
        else:
            # Check local agents
            agents = {
                "Policy Analysis": backend_settings.LOCAL_POLICY_AGENT_URL,
                "Application Processing": backend_settings.LOCAL_APPLICATION_AGENT_URL,
                "Decision Making": backend_settings.LOCAL_DECISION_AGENT_URL
            }
        
        # Check each agent
        for agent_name, agent_url in agents.items():
            is_healthy, status_message = await self.check_agent_health(agent_url)
            results[agent_name] = {
                "url": agent_url,
                "healthy": is_healthy,
                "status": status_message,
                "type": "deployed" if backend_settings.USE_DEPLOYED_AGENTS else "local"
            }
            
        return results
    
    def start_local_agents(self) -> bool:
        """
        Start local agents using the START_AGENTS.bat script.
        
        Returns:
            True if start command was executed, False otherwise
        """
        try:
            batch_file = self.system_root / "START_AGENTS.bat"
            if not batch_file.exists():
                logger.error(f"START_AGENTS.bat not found at {batch_file}")
                return False
            
            # Execute the batch file
            process = subprocess.Popen(
                str(batch_file),
                cwd=str(self.system_root),
                shell=True,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            
            logger.info(f"Started local agents with PID: {process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start local agents: {e}")
            return False
    
    def stop_local_agents(self) -> bool:
        """
        Stop local agents by killing processes on the agent ports.
        
        Returns:
            True if stop commands were executed, False otherwise
        """
        try:
            ports = [10001, 10002, 10003]
            stopped_any = False
            
            for port in ports:
                try:
                    # Kill process on port (Windows)
                    result = subprocess.run([
                        "netstat", "-ano"
                    ], capture_output=True, text=True, shell=True)
                    
                    for line in result.stdout.split('\n'):
                        if f":{port}" in line and "LISTENING" in line:
                            parts = line.split()
                            if len(parts) >= 5:
                                pid = parts[-1]
                                result = subprocess.run([
                                    "taskkill", "/PID", pid, "/F"
                                ], shell=True, capture_output=True, text=True)
                                stopped_any = True
                                logger.info(f"Stopped process on port {port} (PID: {pid})")
                                
                except Exception as e:
                    logger.warning(f"Failed to stop process on port {port}: {e}")
            
            return stopped_any
            
        except Exception as e:
            logger.error(f"Failed to stop local agents: {e}")
            return False
    
    async def get_agent_stats(self) -> Dict:
        """
        Get comprehensive agent statistics.
        
        Returns:
            Dict with agent statistics and system info
        """
        agent_health = await self.check_all_agents_health()
        
        healthy_count = sum(1 for agent in agent_health.values() if agent["healthy"])
        total_count = len(agent_health)
        
        return {
            "mode": "Deployed" if backend_settings.USE_DEPLOYED_AGENTS else "Local",
            "total_agents": total_count,
            "healthy_agents": healthy_count,
            "unhealthy_agents": total_count - healthy_count,
            "all_healthy": healthy_count == total_count,
            "agent_details": agent_health
        }

    def get_agent_mode(self) -> str:
        """Get current agent mode."""
        return "deployed" if backend_settings.USE_DEPLOYED_AGENTS else "local"
    
    def set_agent_mode(self, mode: str) -> bool:
        """
        Set agent mode and update environment.
        
        Args:
            mode: "local" or "deployed"
            
        Returns:
            True if mode was set successfully
        """
        try:
            if mode not in ["local", "deployed"]:
                return False
            
            # Update environment file
            env_file = self.system_root / ".env"
            if env_file.exists():
                content = env_file.read_text()
                
                # Update AGENT_MODE
                lines = content.split('\n')
                updated_lines = []
                found_agent_mode = False
                
                for line in lines:
                    if line.startswith('AGENT_MODE='):
                        updated_lines.append(f'AGENT_MODE={mode}')
                        found_agent_mode = True
                    else:
                        updated_lines.append(line)
                
                if not found_agent_mode:
                    updated_lines.append(f'AGENT_MODE={mode}')
                
                env_file.write_text('\n'.join(updated_lines))
            
            # Update backend settings
            backend_settings.AGENT_MODE = mode
            backend_settings.USE_DEPLOYED_AGENTS = (mode == "deployed")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set agent mode: {e}")
            return False


# Global instance
agent_manager = AgentManager()