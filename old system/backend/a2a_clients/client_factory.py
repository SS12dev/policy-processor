"""Factory for creating A2A clients based on configuration."""

import logging
from typing import Union

import sys
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(current_dir))

from policy_client import PolicyAnalysisClient
from application_client import ApplicationProcessingClient
from decision_client import DecisionMakingClient
from deployed_client import (
    DeployedPolicyAnalysisClient,
    DeployedApplicationProcessingClient,  
    DeployedDecisionMakingClient
)
from settings import backend_settings

logger = logging.getLogger(__name__)


class ClientFactory:
    """Factory for creating appropriate A2A clients based on configuration."""

    @staticmethod
    def create_policy_client() -> Union[PolicyAnalysisClient, DeployedPolicyAnalysisClient]:
        """
        Create policy analysis client based on configuration.
        
        Returns:
            Policy analysis client (local or deployed)
        """
        if backend_settings.USE_DEPLOYED_AGENTS:
            logger.info("Creating deployed policy analysis client")
            return DeployedPolicyAnalysisClient(backend_settings.DEPLOYED_POLICY_AGENT_URL)
        else:
            logger.info("Creating local policy analysis client")
            return PolicyAnalysisClient(backend_settings.LOCAL_POLICY_AGENT_URL)

    @staticmethod
    def create_application_client() -> Union[ApplicationProcessingClient, DeployedApplicationProcessingClient]:
        """
        Create application processing client based on configuration.
        
        Returns:
            Application processing client (local or deployed)
        """
        if backend_settings.USE_DEPLOYED_AGENTS:
            logger.info("Creating deployed application processing client")
            return DeployedApplicationProcessingClient(backend_settings.DEPLOYED_APPLICATION_AGENT_URL)
        else:
            logger.info("Creating local application processing client")
            return ApplicationProcessingClient(backend_settings.LOCAL_APPLICATION_AGENT_URL)

    @staticmethod
    def create_decision_client() -> Union[DecisionMakingClient, DeployedDecisionMakingClient]:
        """
        Create decision making client based on configuration.
        
        Returns:
            Decision making client (local or deployed)
        """
        if backend_settings.USE_DEPLOYED_AGENTS:
            logger.info("Creating deployed decision making client")
            return DeployedDecisionMakingClient(backend_settings.DEPLOYED_DECISION_AGENT_URL)
        else:
            logger.info("Creating local decision making client")
            return DecisionMakingClient(backend_settings.LOCAL_DECISION_AGENT_URL)

    @staticmethod
    def get_client_mode() -> str:
        """
        Get current client mode.
        
        Returns:
            'deployed' or 'local'
        """
        return "deployed" if backend_settings.USE_DEPLOYED_AGENTS else "local"