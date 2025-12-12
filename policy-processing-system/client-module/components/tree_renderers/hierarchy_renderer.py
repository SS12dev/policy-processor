"""
Policy Hierarchy Renderer

This module provides visualization for policy hierarchies showing:
- Parent-child relationships
- Root policies and sub-policies
- Hierarchical levels with indentation
- Policy metadata (conditions, descriptions)
"""

import streamlit as st
from typing import Dict, List, Any, Optional


def display_policy_hierarchy(hierarchy: dict):
    """
    Display policy hierarchy with visualization and clear parent-child relationships.
    
    Args:
        hierarchy: Dictionary containing hierarchy data with keys:
            - total_policies: Total number of policies
            - total_root_policies: Number of root-level policies
            - max_depth: Maximum depth of the hierarchy
            - root_policies: List of root policy dictionaries
    """
    st.markdown("### [HIERARCHY] Policy Hierarchy Structure")
    
    # Overview metrics - calculate from actual data to avoid backend bugs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Policies", hierarchy.get("total_policies", 0))
    with col2:
        # Calculate root policies from actual data (backend sometimes returns 0 incorrectly)
        total_roots = len(hierarchy.get("root_policies", [])) if "root_policies" in hierarchy else hierarchy.get("total_root_policies", 0)
        st.metric("Root Policies", total_roots, help="Top-level policies that don't depend on others")
    with col3:
        max_depth = hierarchy.get("max_depth", 0)
        st.metric("Max Depth", max_depth, help="Maximum levels in the hierarchy")
    with col4:
        # Calculate total child policies
        total_children = hierarchy.get("total_policies", 0) - total_roots
        st.metric("Sub-Policies", total_children, help="Policies that are children of root policies")
    
    st.markdown("---")
    
    # Display explanation
    with st.expander("[INFO] **Understanding the Hierarchy**", expanded=False):
        st.markdown("""
        **Policy Hierarchy** organizes policies into parent-child relationships:
        
        - **Root Policies (Level 0)**: Top-level coverage rules that stand alone
          - These are the main policies an application is evaluated against
          
        - **Sub-Policies (Level 1+)**: Specific eligibility criteria under root policies
          - These provide detailed conditions and requirements
          
        **How Applications Are Evaluated:**
        1. System identifies applicable root policies based on application type
        2. For each root policy, the decision tree navigates through questions
        3. Sub-policies are checked for additional specific criteria
        4. Final decision is made by aggregating all policy results
        """)
    
    st.markdown("---")
    
    # Display root policies with visuals
    if "root_policies" in hierarchy and hierarchy["root_policies"]:
        for idx, policy in enumerate(hierarchy["root_policies"], 1):
            render_policy_card(policy, idx)
    else:
        st.warning("[WARNING] No root policies found in hierarchy")


def render_policy_card(policy: dict, idx: int):
    """
    Render a single policy as a card with its children.
    
    Args:
        policy: Policy dictionary containing:
            - title: Policy title
            - description: Policy description
            - policy_id: Unique identifier
            - level: Hierarchy level
            - conditions: List of conditions
            - children: List of child policies
        idx: Index for display numbering
    """
    # Create a card-like container for each root policy
    with st.container():
        # Header with icon and level badge
        col_header, col_badge = st.columns([5, 1])
        with col_header:
            st.markdown(f"### [ROOT] Root Policy {idx}: {policy.get('title', 'Untitled Policy')}")
        with col_badge:
            st.markdown(
                f"<div style='text-align: right; padding: 8px; background-color: #1f77b4; "
                f"color: white; border-radius: 5px; font-weight: bold;'>Level {policy.get('level', 0)}</div>",
                unsafe_allow_html=True
            )
        
        # Policy details
        st.markdown(f"**Description:** {policy.get('description', 'No description')}")
        st.markdown(f"**Policy ID:** `{policy.get('policy_id', 'N/A')}`")
        
        # Conditions
        if policy.get("conditions"):
            with st.expander(f"[CONDITIONS] View Conditions ({len(policy['conditions'])} total)", expanded=False):
                for cond_idx, cond in enumerate(policy["conditions"], 1):
                    st.markdown(f"{cond_idx}. {cond.get('description', 'No description')}")
        
        # Display children (sub-policies) with indentation
        children = policy.get("children", [])
        if children:
            st.markdown(f"**Sub-Policies:** {len(children)} child policies")
            
            # Show children in an indented section
            for child_idx, child in enumerate(children, 1):
                render_child_policy(child, child_idx)
        else:
            st.markdown("**Sub-Policies:** No child policies (leaf node)")
        
        st.markdown("---")


def render_child_policy(child: dict, child_idx: int):
    """
    Render a child (sub) policy with indentation.
    
    Args:
        child: Child policy dictionary
        child_idx: Index for display numbering
    """
    with st.container():
        # Indented child display
        st.markdown(f"""
        <div style='margin-left: 40px; padding: 15px; background-color: #f0f2f6; 
                    border-left: 4px solid #4CAF50; border-radius: 5px; margin-bottom: 10px;'>
            <strong>|-- Sub-Policy {child_idx}: {child.get('title', 'Untitled')}</strong><br/>
            <span style='color: #666;'>{child.get('description', 'No description')}</span><br/>
            <span style='color: #888; font-size: 0.9em;'>Level {child.get('level', 1)} | Policy ID: {child.get('policy_id', 'N/A')}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Show child conditions
        if child.get("conditions"):
            with st.expander(f"    [CONDITIONS] Sub-Policy Conditions ({len(child['conditions'])} total)", expanded=False):
                for cond_idx, cond in enumerate(child["conditions"], 1):
                    st.markdown(f"    {cond_idx}. {cond.get('description', 'No description')}")
        
        # Recursively show grandchildren if they exist
        if child.get("children"):
            st.markdown(f"    **|-- Nested Sub-Policies:** {len(child['children'])} grandchildren")
            for grandchild in child["children"]:
                render_grandchild_policy(grandchild)


def render_grandchild_policy(grandchild: dict):
    """
    Render a grandchild policy (nested sub-policy).
    
    Args:
        grandchild: Grandchild policy dictionary
    """
    st.markdown(f"""
    <div style='margin-left: 80px; padding: 10px; background-color: #e8f5e9; 
                border-left: 3px solid #66BB6A; border-radius: 5px; margin-bottom: 8px; font-size: 0.9em;'>
        <strong>|-- {grandchild.get('title', 'Untitled')}</strong><br/>
        <span style='color: #666;'>{grandchild.get('description', 'No description')}</span><br/>
        <span style='color: #888; font-size: 0.85em;'>Level {grandchild.get('level', 2)}</span>
    </div>
    """, unsafe_allow_html=True)
