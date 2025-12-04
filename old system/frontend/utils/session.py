"""
Session utilities for Streamlit components
"""
import streamlit as st
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from backend.a2a_clients.orchestrator import PriorAuthOrchestrator


def get_orchestrator():
    """Get or create orchestrator instance"""
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = PriorAuthOrchestrator()
    return st.session_state.orchestrator