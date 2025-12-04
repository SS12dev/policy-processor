"""
Prior Authorization System - Modular Frontend
"""
import streamlit as st
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database.db_connection import init_db
from frontend.components.agent_status import show_agent_status
from frontend.components.policy_management import show_policy_management
from frontend.components.application_processing import show_application_processing
from frontend.components.decision_review import show_decision_review

st.set_page_config(
    page_title="Prior Authorization System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'current_policy_id' not in st.session_state:
    st.session_state.current_policy_id = None
if 'current_questionnaire_id' not in st.session_state:
    st.session_state.current_questionnaire_id = None
if 'current_application_id' not in st.session_state:
    st.session_state.current_application_id = None


def main():
    """Main application entry point"""
    st.title("Prior Authorization System")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Agent Status", "Policy Management", "Application Processing", "Decision Review"]
    )
    
    # Route to appropriate page
    if page == "Agent Status":
        show_agent_status()
    elif page == "Policy Management":
        show_policy_management()
    elif page == "Application Processing":
        show_application_processing()
    elif page == "Decision Review":
        show_decision_review()


if __name__ == "__main__":
    # Initialize database on first run only
    if 'db_initialized' not in st.session_state:
        init_db()
        st.session_state.db_initialized = True
    main()