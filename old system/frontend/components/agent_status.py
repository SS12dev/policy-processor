"""Agent Status Management Component for Streamlit."""

import asyncio
import streamlit as st
import time
from datetime import datetime
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from frontend.utils.agent_manager import agent_manager


def show_agent_status():
    """Display comprehensive agent status management interface."""
    
    st.header("Agent Status & Management")
    st.markdown("Monitor and manage both local and deployed agents for the Prior Authorization system.")
    
    # Agent Mode Selection
    st.subheader("Agent Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        current_mode = agent_manager.get_agent_mode()
        
        mode_options = {
            "Local Agents": "local",
            "Deployed Agents": "deployed"
        }
        
        selected_mode_label = "Local Agents" if current_mode == "local" else "Deployed Agents"
        
        new_mode = st.selectbox(
            "Select Agent Mode",
            options=list(mode_options.keys()),
            index=list(mode_options.keys()).index(selected_mode_label),
            help="Choose between local development agents or deployed production agents"
        )
        
        if mode_options[new_mode] != current_mode:
            if st.button("Apply Agent Mode Change", type="primary"):
                if agent_manager.set_agent_mode(mode_options[new_mode]):
                    st.success(f"Agent mode changed to: {new_mode}")
                    st.rerun()
                else:
                    st.error("Failed to change agent mode")
    
    with col2:
        st.info(f"**Current Mode:** {current_mode.title()}")
        if st.button("Refresh Status", type="secondary"):
            st.rerun()
    
    st.divider()
    
    # Agent Health Status
    st.subheader("Agent Health Status")
    
    # Get agent stats asynchronously
    with st.spinner("Checking agent health..."):
        try:
            # Run async function in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            agent_stats = loop.run_until_complete(agent_manager.get_agent_stats())
            loop.close()
            
        except Exception as e:
            st.error(f"Failed to get agent status: {str(e)}")
            return
    
    # Display overall status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Agent Mode", 
            agent_stats["mode"],
            help="Current agent deployment mode"
        )
    
    with col2:
        st.metric(
            "Total Agents", 
            agent_stats["total_agents"],
            help="Total number of agents in system"
        )
    
    with col3:
        st.metric(
            "Healthy Agents", 
            agent_stats["healthy_agents"],
            delta=agent_stats["healthy_agents"] - agent_stats["unhealthy_agents"] if agent_stats["unhealthy_agents"] > 0 else None,
            help="Number of agents responding to health checks"
        )
    
    with col4:
        overall_status = "Healthy" if agent_stats["all_healthy"] else "Issues Detected"
        st.metric(
            "System Status", 
            overall_status,
            help="Overall system health status"
        )
    
    # Display individual agent status
    st.subheader("Individual Agent Status")
    
    for agent_name, agent_info in agent_stats["agent_details"].items():
        with st.expander(f"{agent_name} Agent", expanded=not agent_info["healthy"]):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**URL:** `{agent_info['url']}`")
                st.write(f"**Type:** {agent_info['type'].title()}")
            
            with col2:
                if agent_info["healthy"]:
                    st.success("Healthy")
                else:
                    st.error("Unhealthy")
            
            with col3:
                st.write(f"**Status:** {agent_info['status']}")
    
    # Local Agent Management (only show if in local mode)
    if current_mode == "local":
        st.divider()
        st.subheader("Local Agent Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Start Local Agents", type="primary", use_container_width=True):
                with st.spinner("Starting local agents..."):
                    if agent_manager.start_local_agents():
                        st.success("Local agents start command executed. Please wait a few seconds and refresh status.")
                        st.info("Agents are starting in separate console windows. This may take 10-30 seconds.")
                    else:
                        st.error("Failed to start local agents. Check that START_AGENTS.bat exists.")
        
        with col2:
            if st.button("Stop Local Agents", type="secondary", use_container_width=True):
                with st.spinner("Stopping local agents..."):
                    if agent_manager.stop_local_agents():
                        st.success("Local agents stop command executed.")
                    else:
                        st.warning("No local agents found to stop.")
        
        st.info("""
        **Local Agent Management:**
        - **Start**: Launches agents in separate console windows on ports 10001, 10002, 10003
        - **Stop**: Terminates agent processes running on those ports
        - Agents may take 10-30 seconds to fully initialize after starting
        """)
    
    # Deployed Agent Information (only show if in deployed mode)
    elif current_mode == "deployed":
        st.divider()
        st.subheader("Deployed Agent Information")
        
        st.info("""
        **Deployed Agent Details:**
        - Agents are running in the DI environment
        - No local management required
        - Health status shows real-time connectivity
        - Contact system administrator if agents are unhealthy
        """)
    
    # Auto-refresh option
    st.divider()
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
    
    with col2:
        if auto_refresh:
            st.info("Page will auto-refresh every 30 seconds")
            time.sleep(30)
            st.rerun()
    
    # Last updated timestamp
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    show_agent_status()