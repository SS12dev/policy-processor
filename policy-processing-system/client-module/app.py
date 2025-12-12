"""
Streamlit Application for Policy Document Processor.

Two-tab interface:
1. Upload & Process - Upload policy documents and process them
2. Review Decision Trees - View and interact with generated decision trees
"""

import streamlit as st
import json
import base64
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import app modules
from a2a_client import A2AClientSync
from backend_handler import UIBackendHandler
from database.operations import DatabaseOperations
from utils.logger import get_logger
from components import render_tile_tree_view, render_tree_selector
from components.tree_renderers import (
    display_decision_tree,
    display_policy_hierarchy
)

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Policy Document Processor",
    page_icon=":page_facing_up:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database operations
@st.cache_resource
def get_database():
    """Get database operations instance."""
    db_path = Path("./data/policy_processor.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return DatabaseOperations(database_url=f"sqlite:///{db_path}")

# Initialize A2A client
@st.cache_resource
def get_a2a_client():
    """Get A2A client instance."""
    return A2AClientSync(base_url="http://localhost:8001", timeout=300.0)

# Initialize backend handler
@st.cache_resource
def get_backend_handler():
    """Get backend handler instance."""
    return UIBackendHandler(get_database())

db_ops = get_database()
a2a_client = get_a2a_client()
backend_handler = get_backend_handler()

# App header
st.title("Policy Document Processor")
st.markdown("---")

# Sidebar - Simplified Status
with st.sidebar:
    st.header("Status")

    # Check A2A server health
    if a2a_client.check_health():
        st.success("● Server Online")
    else:
        st.error("● Server Offline")

    # Total policies count
    total_jobs = db_ops.count_jobs(status="completed")
    st.metric("Total Policies", total_jobs)
    
    # View all button
    if st.button("View All Policies", use_container_width=True):
        st.session_state.active_tab = 1


# ============================================================================
# REMOVED DEPRECATED FUNCTIONS - All duplicates removed
# Only the active _display_decision_tree_interactive and helper functions remain
# ============================================================================


# ============================================================================
# Decision Tree Display - Current Active Version
# ============================================================================

def _display_decision_tree_interactive(tree: dict, tree_idx: int):
    """Display decision tree in a clean tile-based vertical format with AND/OR grouping."""
    
    policy_title = tree.get('policy_title', 'Untitled Policy')
    policy_id = tree.get('policy_id', 'N/A')
    
    # Tree header with gradient and better contrast
    st.markdown(f"""
    <div style='padding: 25px; background: linear-gradient(135deg, #4A5ED7 0%, #5B4DB2 100%); 
                border-radius: 12px; margin-bottom: 25px; color: white; 
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <h2 style='margin: 0; color: white; font-weight: 600;'>🌳 {policy_title}</h2>
        <p style='margin: 8px 0 0 0; color: rgba(255,255,255,0.95); font-size: 0.95em;'>
            <strong>Policy ID:</strong> {policy_id}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get questions and root node
    questions = tree.get("questions", [])
    tree_root = tree.get('tree', {}).get('root', {}) if 'tree' in tree else {}
    
    if not questions and not tree_root:
        st.warning("No questions or tree structure found")
        return
    
    # Display tile-based tree structure
    _render_tree_tiles(tree_root, questions, tree_idx)


def _render_tree_tiles(root_node: dict, questions: list, tree_idx: int):
    """Render decision tree as vertical tiles with AND/OR grouping."""
    
    if not root_node and not questions:
        st.info("No decision tree structure available")
        return
    
    # If we have questions list, display them with path visualization
    if questions:
        st.markdown("### 📊 Decision Flow")
        
        for q_idx, question in enumerate(questions, 1):
            question_text = question.get('question_text', question.get('text', 'No question'))
            question_id = question.get('question_id', question.get('id', f'q{q_idx}'))
            answer_type = question.get('answer_type', question.get('type', 'unknown'))
            
            # Question tile with better contrast and indentation marker
            st.markdown(f"""
            <div style='padding: 20px; background-color: #1E3A8A; border-left: 6px solid #3B82F6; 
                        border-radius: 10px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <div style='display: flex; align-items: center; margin-bottom: 12px;'>
                    <span style='background: #3B82F6; color: white; padding: 4px 12px; 
                                 border-radius: 20px; font-weight: 600; font-size: 0.85em;'>
                        QUESTION {q_idx}
                    </span>
                </div>
                <h3 style='margin: 0 0 10px 0; color: #FFFFFF; font-weight: 500; line-height: 1.4;'>
                    ❓ {question_text}
                </h3>
                <p style='color: #93C5FD; margin: 0; font-size: 0.9em;'>
                    <strong>Answer Type:</strong> <code style='background: rgba(255,255,255,0.1); 
                    padding: 2px 8px; border-radius: 4px; color: #DBEAFE;'>{answer_type}</code>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Get branches/conditions
            branches = question.get('branches', [])
            conditions = question.get('conditions', [])
            routing_rules = question.get('routing_rules', [])
            next_question_logic = question.get('next_question_logic', {})
            
            # Display branches in tiles
            if branches:
                _render_branches_as_tiles(branches, q_idx, tree_idx)
            elif conditions:
                _render_conditions_as_tiles(conditions, q_idx)
            elif routing_rules:
                _render_routing_rules_as_tiles(routing_rules, q_idx)
            elif next_question_logic:
                _render_next_question_logic_as_tiles(next_question_logic, q_idx)
            else:
                # Try to find branches in nested structure
                if 'children' in question or 'outcomes' in question:
                    children = question.get('children', question.get('outcomes', []))
                    if children:
                        _render_nested_outcomes(children, q_idx)
                    else:
                        st.caption("🔍 No branch information available for this question")
    
    # If we have tree root, render it as hierarchical tiles
    elif root_node:
        _render_node_hierarchy(root_node, level=0, tree_idx=tree_idx)


def _render_branches_as_tiles(branches: list, question_idx: int, tree_idx: int):
    """Render branches as vertical tiles with AND/OR grouping."""
    
    st.markdown("#### 🔀 Answer Paths:")
    
    # Group branches by logic operator if present
    and_branches = []
    or_branches = []
    simple_branches = []
    
    for branch in branches:
        logic_op = branch.get('logic_operator', branch.get('operator', '')).upper()
        if logic_op == 'AND':
            and_branches.append(branch)
        elif logic_op == 'OR':
            or_branches.append(branch)
        else:
            simple_branches.append(branch)
    
    # Display AND conditions group
    if and_branches:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #065F46 0%, #047857 100%); 
                    padding: 15px 20px; border-radius: 10px; margin: 20px 0 15px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <div style='color: #FFFFFF; font-weight: 600; font-size: 1em;'>
                🔗 AND CONDITIONS - All Must Be Satisfied
            </div>
            <div style='color: #D1FAE5; font-size: 0.85em; margin-top: 4px;'>
                Every condition below must be true to proceed
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        for idx, branch in enumerate(and_branches, 1):
            _render_single_branch_tile(branch, tree_idx, indent_level=1, branch_num=idx)
    
    # Display OR conditions group
    if or_branches:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #C2410C 0%, #EA580C 100%); 
                    padding: 15px 20px; border-radius: 10px; margin: 20px 0 15px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <div style='color: #FFFFFF; font-weight: 600; font-size: 1em;'>
                🔀 OR CONDITIONS - Any One Can Be Satisfied
            </div>
            <div style='color: #FED7AA; font-size: 0.85em; margin-top: 4px;'>
                Only one of the conditions below needs to be true
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        for idx, branch in enumerate(or_branches, 1):
            _render_single_branch_tile(branch, tree_idx, indent_level=1, branch_num=idx)
    
    # Display simple branches
    if simple_branches:
        for idx, branch in enumerate(simple_branches, 1):
            _render_single_branch_tile(branch, tree_idx, indent_level=0, branch_num=idx)


def _render_single_branch_tile(branch: dict, tree_idx: int, indent_level: int = 0, branch_num: int = 1):
    """Render a single branch as a tile with improved contrast and path visualization."""
    
    condition = branch.get('condition', branch.get('answer', branch.get('value', 'N/A')))
    next_step = branch.get('next_question', branch.get('next_node', branch.get('next', '')))
    outcome = branch.get('outcome', branch.get('result', {}))
    description = branch.get('description', branch.get('desc', ''))
    
    # Calculate indentation
    indent = indent_level * 30
    
    # Determine if this is an outcome or continuation
    is_outcome = outcome or branch.get('is_terminal', False) or branch.get('outcome_type')
    
    if is_outcome:
        # This is a terminal outcome
        outcome_type = branch.get('outcome_type', outcome.get('type', 'unknown') if isinstance(outcome, dict) else 'unknown')
        outcome_label = branch.get('outcome_label', outcome.get('label', 'Decision') if isinstance(outcome, dict) else str(outcome))
        outcome_desc = description or (outcome.get('description', '') if isinstance(outcome, dict) else '')
        
        # Color coding with better contrast
        if 'approv' in outcome_type.lower() or 'approv' in outcome_label.lower():
            bg_color = '#065F46'  # Dark green
            border_color = '#10B981'  # Bright green
            icon = '✅'
            text_color = '#FFFFFF'
        elif 'den' in outcome_type.lower() or 'den' in outcome_label.lower():
            bg_color = '#991B1B'  # Dark red
            border_color = '#EF4444'  # Bright red
            icon = '❌'
            text_color = '#FFFFFF'
        elif 'review' in outcome_type.lower() or 'refer' in outcome_type.lower():
            bg_color = '#92400E'  # Dark amber
            border_color = '#F59E0B'  # Bright amber
            icon = '⏸️'
            text_color = '#FFFFFF'
        else:
            bg_color = '#6B21A8'  # Dark purple
            border_color = '#A855F7'  # Bright purple
            icon = '📋'
            text_color = '#FFFFFF'
        
        st.markdown(f"""
        <div style='margin-left: {indent}px; border-left: {f"3px solid #94A3B8; padding-left: 15px;" if indent_level > 0 else ""} margin-bottom: 15px;'>
            <div style='padding: 20px; background-color: {bg_color}; 
                        border-left: 6px solid {border_color}; border-radius: 10px; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 12px;'>
                    <span style='background: {border_color}; color: #1F2937; padding: 4px 12px; 
                                 border-radius: 20px; font-weight: 600; font-size: 0.8em;'>
                        PATH {branch_num}
                    </span>
                </div>
                <div style='font-size: 0.95em; color: #E5E7EB; margin-bottom: 12px;'>
                    <div style='display: flex; align-items: center; gap: 8px;'>
                        <span style='background: rgba(255,255,255,0.15); padding: 4px 10px; 
                                     border-radius: 4px; font-weight: 600; font-size: 0.85em;'>IF</span>
                        <code style='background: rgba(255,255,255,0.1); padding: 6px 12px; 
                                     border-radius: 6px; color: #FCD34D; font-size: 0.9em;'>{condition}</code>
                    </div>
                </div>
                <div style='display: flex; align-items: center; gap: 10px; margin-top: 12px;'>
                    <span style='font-size: 1.5em; color: {border_color};'>↓</span>
                    <div style='flex: 1;'>
                        <div style='font-size: 1.3em; color: {text_color}; font-weight: 600;'>
                            {icon} {outcome_label}
                        </div>
                        {f"<div style='color: #D1D5DB; font-size: 0.9em; margin-top: 8px; padding-left: 5px;'>{outcome_desc}</div>" if outcome_desc else ""}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # This continues to another question
        st.markdown(f"""
        <div style='margin-left: {indent}px; border-left: {f"3px solid #94A3B8; padding-left: 15px;" if indent_level > 0 else ""} margin-bottom: 15px;'>
            <div style='padding: 20px; background-color: #374151; 
                        border-left: 6px solid #F97316; border-radius: 10px; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 12px;'>
                    <span style='background: #F97316; color: white; padding: 4px 12px; 
                                 border-radius: 20px; font-weight: 600; font-size: 0.8em;'>
                        PATH {branch_num}
                    </span>
                </div>
                <div style='font-size: 0.95em; color: #E5E7EB; margin-bottom: 12px;'>
                    <div style='display: flex; align-items: center; gap: 8px;'>
                        <span style='background: rgba(255,255,255,0.15); padding: 4px 10px; 
                                     border-radius: 4px; font-weight: 600; font-size: 0.85em;'>IF</span>
                        <code style='background: rgba(255,255,255,0.1); padding: 6px 12px; 
                                     border-radius: 6px; color: #FCD34D; font-size: 0.9em;'>{condition}</code>
                    </div>
                </div>
                <div style='display: flex; align-items: center; gap: 10px; margin-top: 12px;'>
                    <span style='font-size: 1.5em; color: #F97316;'>→</span>
                    <div style='color: #FFFFFF; font-weight: 500;'>
                        <strong>CONTINUE TO:</strong> <span style='color: #FCD34D;'>{next_step or 'Next question'}</span>
                    </div>
                </div>
                {f"<div style='color: #D1D5DB; font-size: 0.9em; margin-top: 12px; padding: 12px; background: rgba(0,0,0,0.2); border-radius: 6px;'>{description}</div>" if description else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)


def _render_conditions_as_tiles(conditions: list, question_idx: int):
    """Render conditions as tiles."""
    st.markdown("#### Conditions:")
    
    for idx, condition in enumerate(conditions, 1):
        if isinstance(condition, str):
            st.markdown(f"""
            <div style='padding: 12px; background-color: #f0f0f0; border-left: 4px solid #757575; 
                        border-radius: 5px; margin: 8px 0 8px 20px;'>
                <strong>{idx}.</strong> {condition}
            </div>
            """, unsafe_allow_html=True)
        elif isinstance(condition, dict):
            cond_text = condition.get('description', condition.get('text', condition.get('condition', 'N/A')))
            st.markdown(f"""
            <div style='padding: 12px; background-color: #f0f0f0; border-left: 4px solid #757575; 
                        border-radius: 5px; margin: 8px 0 8px 20px;'>
                <strong>{idx}.</strong> {cond_text}
            </div>
            """, unsafe_allow_html=True)


def _render_routing_rules_as_tiles(routing_rules: list, question_idx: int):
    """Render routing rules as tiles."""
    st.markdown("#### Routing Rules:")
    
    for rule in routing_rules:
        condition = rule.get('condition', rule.get('if', 'N/A'))
        destination = rule.get('destination', rule.get('then', rule.get('next', 'N/A')))
        
        st.markdown(f"""
        <div style='padding: 15px; background-color: #e3f2fd; border-left: 5px solid #2196F3; 
                    border-radius: 8px; margin: 10px 0 10px 20px;'>
            <div style='color: #555; margin-bottom: 5px;'><strong>If:</strong> {condition}</div>
            <div style='color: #1976D2; font-weight: 500;'>→ Then: {destination}</div>
        </div>
        """, unsafe_allow_html=True)


def _render_next_question_logic_as_tiles(logic: dict, question_idx: int):
    """Render next question logic as tiles."""
    st.markdown("#### Decision Logic:")
    
    for key, value in logic.items():
        st.markdown(f"""
        <div style='padding: 15px; background-color: #fff3e0; border-left: 5px solid #FF9800; 
                    border-radius: 8px; margin: 10px 0 10px 20px;'>
            <div style='color: #555;'><strong>{key}:</strong></div>
            <div style='color: #e65100; margin-top: 5px;'>{value}</div>
        </div>
        """, unsafe_allow_html=True)


def _render_nested_outcomes(children: list, question_idx: int):
    """Render nested outcomes/children."""
    st.markdown("#### Possible Outcomes:")
    
    for child in children:
        if isinstance(child, dict):
            outcome_type = child.get('type', child.get('outcome_type', 'outcome'))
            label = child.get('label', child.get('title', 'Outcome'))
            description = child.get('description', child.get('desc', ''))
            
            # Color based on type
            if 'approv' in outcome_type.lower() or 'approv' in label.lower():
                bg_color = '#c8e6c9'
                border_color = '#4CAF50'
                icon = '✅'
            elif 'den' in outcome_type.lower() or 'den' in label.lower():
                bg_color = '#ffcdd2'
                border_color = '#f44336'
                icon = '❌'
            else:
                bg_color = '#fff9c4'
                border_color = '#FFC107'
                icon = '📋'
            
            st.markdown(f"""
            <div style='padding: 15px; background-color: {bg_color}; 
                        border-left: 5px solid {border_color}; border-radius: 8px; 
                        margin: 10px 0 10px 20px;'>
                <div style='font-size: 1.2em; margin-bottom: 8px;'>{icon} <strong>{label}</strong></div>
                {f"<div style='color: #666; font-size: 0.95em;'>{description}</div>" if description else ""}
            </div>
            """, unsafe_allow_html=True)


def _render_node_hierarchy(node: dict, level: int, tree_idx: int):
    """Render node hierarchy recursively as tiles."""
    
    if not node:
        return
    
    indent = level * 30
    node_type = node.get('type', 'unknown')
    
    # Render based on node type
    if node_type == 'question':
        question_text = node.get('question', node.get('text', 'No question'))
        
        st.markdown(f"""
        <div style='margin-left: {indent}px; padding: 15px; background-color: #e3f2fd; 
                    border-left: 4px solid #2196F3; border-radius: 8px; margin-bottom: 12px;'>
            <div style='color: #1976D2; font-weight: 600;'>❓ {question_text}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Render children (yes/no branches)
        for answer_key in ['yes', 'no', 'true', 'false']:
            if answer_key in node:
                st.markdown(f"""
                <div style='margin-left: {indent + 20}px; color: #666; font-size: 0.9em; margin: 5px 0;'>
                    <strong>If {answer_key.upper()}:</strong>
                </div>
                """, unsafe_allow_html=True)
                _render_node_hierarchy(node[answer_key], level + 1, tree_idx)
        
        # Check for generic children list
        if 'children' in node:
            for child in node['children']:
                _render_node_hierarchy(child, level + 1, tree_idx)
    
    elif node_type == 'outcome':
        outcome_label = node.get('label', node.get('outcome', 'Decision'))
        outcome_type = node.get('outcome_type', 'unknown')
        description = node.get('description', '')
        
        # Color coding
        if 'approv' in outcome_type.lower():
            bg_color = '#c8e6c9'
            border_color = '#4CAF50'
            icon = '✅'
        elif 'den' in outcome_type.lower():
            bg_color = '#ffcdd2'
            border_color = '#f44336'
            icon = '❌'
        else:
            bg_color = '#fff9c4'
            border_color = '#FFC107'
            icon = '📋'
        
        st.markdown(f"""
        <div style='margin-left: {indent}px; padding: 15px; background-color: {bg_color}; 
                    border-left: 5px solid {border_color}; border-radius: 8px; margin-bottom: 12px;'>
            <div style='font-size: 1.15em; margin-bottom: 5px;'>{icon} <strong>{outcome_label}</strong></div>
            {f"<div style='color: #666; font-size: 0.9em;'>{description}</div>" if description else ""}
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # Generic node
        st.markdown(f"""
        <div style='margin-left: {indent}px; padding: 12px; background-color: #f5f5f5; 
                    border-left: 3px solid #9E9E9E; border-radius: 5px; margin-bottom: 10px;'>
            <div style='color: #555;'>📌 {node.get('label', node.get('id', 'Node'))}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Render children if present
        if 'children' in node:
            for child in node['children']:
                _render_node_hierarchy(child, level + 1, tree_idx)


def _display_decision_tree_interactive_old(tree: dict, tree_idx: int):
    """Display an interactive decision tree with branch navigation and decision path visualization."""
    
    policy_title = tree.get('policy_title', 'Untitled Policy')
    policy_id = tree.get('policy_id', 'N/A')
    
    st.markdown(f"### 🌳 Decision Tree {tree_idx}: {policy_title}")
    
    # Tree statistics - calculate from actual data if not provided
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # Calculate from actual questions if total_questions is 0 or missing
        total_q = tree.get('total_questions', 0)
        if total_q == 0 and 'questions' in tree:
            total_q = len(tree['questions'])
        st.metric("Total Questions", total_q)
    with col2:
        max_d = tree.get('max_depth', 0)
        # If max_depth is missing, try to calculate from tree structure
        if max_d == 0 and 'tree' in tree:
            # Try to find max depth from tree structure
            max_d = tree.get('tree', {}).get('max_depth', 0)
        st.metric("Max Depth", max_d)
    with col3:
        total_p = tree.get('total_paths', 0)
        # Alternative key names
        if total_p == 0:
            total_p = tree.get('num_paths', tree.get('path_count', 'N/A'))
        st.metric("Decision Paths", total_p)
    with col4:
        confidence = tree.get('confidence', 0)
        # Handle confidence as percentage or decimal
        if confidence > 1:
            conf_display = f"{confidence:.0f}%"
        elif confidence > 0:
            conf_display = f"{confidence*100:.0f}%"
        else:
            # Try alternative keys
            confidence = tree.get('generation_confidence', tree.get('tree_confidence', 0))
            conf_display = f"{confidence*100:.0f}%" if confidence > 0 else "N/A"
        st.metric("Confidence", conf_display)
    
    st.markdown(f"**Policy ID:** `{policy_id}`")
    
    # Explanation of how this tree is used
    with st.expander("ℹ️ **How This Decision Tree Works**", expanded=False):
        st.markdown(f"""
        **Decision Tree for: {policy_title}**
        
        This decision tree evaluates applications against the "{policy_title}" policy through a series of questions:
        
        1. **Start**: Application begins at the first question
        2. **Navigation**: Based on each answer, the tree branches to the next relevant question
        3. **Outcomes**: Each path through the tree leads to a decision (Approved/Denied/Refer)
        4. **Transparency**: Every decision can be traced back through the exact questions asked
        
        **Interactive Navigation Below:**
        - Select any question to see its branches
        - Follow different answer paths to see outcomes
        - Understand how decisions are made step-by-step
        """)
    
    st.markdown("---")
    
    # Display questions and branches interactively
    questions = tree.get("questions", [])
    
    if questions:
        # Option to view all questions or navigate interactively
        view_mode = st.radio(
            "View Mode:",
            ["🎯 Decision Paths by Outcome", "📋 All Questions List", "🔀 Interactive Navigation"],
            key=f"view_mode_tree_{tree_idx}",
            horizontal=True
        )
        
        if view_mode == "🎯 Decision Paths by Outcome":
            # NEW: Show clear paths to Approval, Denial, and Review
            st.markdown("#### 🎯 Decision Outcomes - Condition-Specific Paths")
            st.info("This view shows the specific conditions that lead to each outcome type")
            
            # Extract all possible outcomes from the tree
            outcomes_map = {"approved": [], "denied": [], "refer": [], "pending": [], "other": []}
            
            # Try to parse tree structure for outcomes
            if 'tree' in tree and 'root' in tree['tree']:
                # Recursively find all outcome nodes
                def find_outcomes(node, path=[]):
                    if not node:
                        return
                    
                    node_type = node.get('type', '')
                    if node_type == 'outcome':
                        outcome_type = node.get('outcome_type', 'other').lower()
                        outcome_info = {
                            'label': node.get('label', 'Unknown'),
                            'description': node.get('description', ''),
                            'path': path.copy(),
                            'conditions': node.get('conditions', [])
                        }
                        if outcome_type in outcomes_map:
                            outcomes_map[outcome_type].append(outcome_info)
                        else:
                            outcomes_map['other'].append(outcome_info)
                    
                    # Traverse children
                    for child_key in ['yes', 'no', 'children', 'branches']:
                        if child_key in node:
                            child = node[child_key]
                            if isinstance(child, dict):
                                new_path = path + [f"→ {node.get('question', 'Next step')}"]
                                find_outcomes(child, new_path)
                            elif isinstance(child, list):
                                for c in child:
                                    if isinstance(c, dict):
                                        new_path = path + [f"→ {node.get('question', 'Next step')}"]
                                        find_outcomes(c, new_path)
                
                find_outcomes(tree['tree'].get('root', {}))
            
            # Display outcomes by type
            tab1, tab2, tab3 = st.tabs(["✅ Approval Paths", "❌ Denial Paths", "⏸️ Review/Pending Paths"])
            
            with tab1:
                st.markdown("### ✅ Conditions Leading to APPROVAL")
                approved = outcomes_map['approved']
                if approved:
                    for idx, outcome in enumerate(approved, 1):
                        st.markdown(f"""
                        <div style='padding: 15px; background-color: #c8e6c9; border-left: 4px solid #4CAF50; border-radius: 5px; margin-bottom: 15px;'>
                            <strong>Approval Path {idx}: {outcome['label']}</strong><br/>
                            <span style='color: #2e7d32; font-size: 0.95em;'>{outcome['description']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if outcome['path']:
                            st.markdown("**Decision Flow:**")
                            for step in outcome['path']:
                                st.markdown(f"  {step}")
                        
                        if outcome['conditions']:
                            st.markdown("**Required Conditions:**")
                            for cond in outcome['conditions']:
                                st.markdown(f"  • {cond}")
                else:
                    st.info("No explicit approval paths found in tree structure. This may indicate the tree uses implicit approvals (when all conditions are met).")
                    
                    # Fallback: Show conditions from policy that would lead to approval
                    st.markdown("**Based on Policy Conditions (Approval Criteria):**")
                    st.markdown(f"""
                    For the "{policy_title}" policy, approval typically requires meeting ALL of these conditions:
                    
                    1. **Primary Eligibility**: Must meet baseline criteria (e.g., BMI thresholds)
                    2. **Medical History**: Documented failed conservative treatment attempts
                    3. **Clinical Assessment**: Completion of required evaluations
                    4. **Safety Criteria**: No contraindications or exclusion factors
                    
                    *Navigate through questions above to see specific requirements*
                    """)
            
            with tab2:
                st.markdown("### ❌ Conditions Leading to DENIAL")
                denied = outcomes_map['denied']
                if denied:
                    for idx, outcome in enumerate(denied, 1):
                        st.markdown(f"""
                        <div style='padding: 15px; background-color: #ffcdd2; border-left: 4px solid #f44336; border-radius: 5px; margin-bottom: 15px;'>
                            <strong>Denial Path {idx}: {outcome['label']}</strong><br/>
                            <span style='color: #c62828; font-size: 0.95em;'>{outcome['description']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if outcome['path']:
                            st.markdown("**Decision Flow:**")
                            for step in outcome['path']:
                                st.markdown(f"  {step}")
                        
                        if outcome['conditions']:
                            st.markdown("**Denial Reasons:**")
                            for cond in outcome['conditions']:
                                st.markdown(f"  • {cond}")
                else:
                    st.info("No explicit denial paths found. Denials typically occur when approval conditions are NOT met.")
                    
                    st.markdown("**Common Denial Reasons:**")
                    st.markdown(f"""
                    Applications for "{policy_title}" are typically denied when:
                    
                    ❌ **Eligibility Criteria Not Met**: BMI below threshold, age restrictions not met  
                    ❌ **Insufficient Medical History**: No documented conservative treatment attempts  
                    ❌ **Missing Requirements**: Required evaluations or assessments not completed  
                    ❌ **Contraindications Present**: Medical or safety concerns that preclude approval  
                    
                    *Review questions above to see specific denial triggers*
                    """)
            
            with tab3:
                st.markdown("### ⏸️ Conditions Requiring REVIEW or Additional Information")
                review = outcomes_map['refer'] + outcomes_map['pending']
                if review:
                    for idx, outcome in enumerate(review, 1):
                        st.markdown(f"""
                        <div style='padding: 15px; background-color: #fff9c4; border-left: 4px solid #FFC107; border-radius: 5px; margin-bottom: 15px;'>
                            <strong>Review Path {idx}: {outcome['label']}</strong><br/>
                            <span style='color: #f57f17; font-size: 0.95em;'>{outcome['description']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if outcome['path']:
                            st.markdown("**Decision Flow:**")
                            for step in outcome['path']:
                                st.markdown(f"  {step}")
                        
                        if outcome['conditions']:
                            st.markdown("**Review Triggers:**")
                            for cond in outcome['conditions']:
                                st.markdown(f"  • {cond}")
                else:
                    st.info("No explicit review/pending paths found.")
                    
                    st.markdown("**Cases Requiring Manual Review:**")
                    st.markdown(f"""
                    For "{policy_title}", manual review may be required when:
                    
                    ⏸️ **Edge Cases**: Applicant meets some but not all criteria (borderline cases)  
                    ⏸️ **Ambiguous Information**: Medical documentation is unclear or incomplete  
                    ⏸️ **Special Circumstances**: Unusual medical history or unique patient factors  
                    ⏸️ **Policy Exceptions**: Request for exception to standard criteria  
                    
                    *These cases are flagged for human clinical judgment*
                    """)
            
            st.markdown("---")
            st.markdown("**💡 Tip:** Use the other view modes below to explore the full decision tree structure and specific questions.")
        
        elif view_mode == "📋 All Questions List":
            # Simple list view
            st.markdown("#### All Questions in This Tree:")
            for q_idx, question in enumerate(questions, 1):
                question_text = question.get('question_text', 'No question')
                answer_type = question.get('answer_type', 'unknown')
                
                with st.expander(f"**Q{q_idx}: {question_text}**", expanded=False):
                    st.markdown(f"**Answer Type:** `{answer_type}`")
                    
                    if answer_type == 'multiple_choice' and question.get('options'):
                        st.markdown("**Answer Options:**")
                        for opt in question['options']:
                            st.markdown(f"- {opt}")
                    elif answer_type == 'yes_no':
                        st.markdown("**Answer Options:** Yes / No")
                    elif answer_type == 'numeric':
                        st.markdown("**Answer Type:** Numeric value")
                    
                    # Show next steps if available
                    if question.get('branches'):
                        st.markdown("**Next Steps Based on Answer:**")
                        for branch in question['branches']:
                            condition = branch.get('condition', 'N/A')
                            next_q = branch.get('next_question', 'End')
                            st.markdown(f"- If {condition} → {next_q}")
        
        elif view_mode == "🔀 Interactive Navigation":
            # Interactive navigation through the tree
            st.markdown("#### Navigate Through the Decision Tree:")
            st.info("👇 Select a question below to explore its branches and see where each answer leads")
            
            selected_question = st.selectbox(
                "Select a Question to Explore:",
                options=range(len(questions)),
                format_func=lambda i: f"Q{i+1}: {questions[i].get('question_text', 'No question')[:80]}...",
                key=f"select_question_tree_{tree_idx}"
            )
            
            if selected_question is not None:
                question = questions[selected_question]
                question_text = question.get('question_text', 'No question')
                answer_type = question.get('answer_type', 'unknown')
                
                # Display selected question details
                st.markdown(f"""
                <div style='padding: 20px; background-color: #e3f2fd; border-left: 5px solid #2196F3; border-radius: 5px; margin-bottom: 20px;'>
                    <h4 style='margin-top: 0; color: #1976D2;'>Question {selected_question + 1}</h4>
                    <p style='font-size: 1.1em; font-weight: 500; margin-bottom: 10px;'>{question_text}</p>
                    <p style='color: #666; margin-bottom: 0;'><strong>Answer Type:</strong> {answer_type}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show answer options and branches
                st.markdown("##### 🔀 Decision Branches:")
                
                branches = question.get('branches', [])
                if branches:
                    for branch_idx, branch in enumerate(branches, 1):
                        condition = branch.get('condition', 'N/A')
                        next_node = branch.get('next_question', 'Decision Outcome')
                        outcome = branch.get('outcome', None)
                        
                        # Determine if this leads to outcome or next question
                        if outcome:
                            # Terminal branch - leads to decision
                            outcome_type = outcome.get('type', 'unknown')
                            outcome_label = outcome.get('label', 'No label')
                            
                            bg_color = '#c8e6c9' if outcome_type == 'approved' else '#ffcdd2' if outcome_type == 'denied' else '#fff9c4'
                            icon = '✅' if outcome_type == 'approved' else '❌' if outcome_type == 'denied' else '⏸️'
                            
                            st.markdown(f"""
                            <div style='padding: 15px; background-color: {bg_color}; border-left: 4px solid #4CAF50; border-radius: 5px; margin-bottom: 10px;'>
                                <strong>Branch {branch_idx}: If answer is "{condition}"</strong><br/>
                                <span style='font-size: 1.2em;'>{icon} <strong>OUTCOME: {outcome_label}</strong></span><br/>
                                <span style='color: #666; font-size: 0.9em;'>This path ends with: {outcome.get('description', 'No description')}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # Continues to next question
                            st.markdown(f"""
                            <div style='padding: 15px; background-color: #f5f5f5; border-left: 4px solid #FF9800; border-radius: 5px; margin-bottom: 10px;'>
                                <strong>Branch {branch_idx}: If answer is "{condition}"</strong><br/>
                                <span style='color: #666;'>→ Continue to: <strong>{next_node}</strong></span>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("No branches defined for this question")
                
                # Show sample decision path
                st.markdown("---")
                st.markdown("##### 📍 Example Decision Path:")
                st.info(f"**Scenario:** Following the first branch from this question would lead to: {branches[0].get('next_question', 'a decision outcome') if branches else 'unknown'}")
    else:
        st.warning("⚠️ No questions found in this decision tree")


def _display_evaluation_flow_guide():
    """Display a comprehensive guide on how applications are evaluated."""
    st.markdown("## 📊 How Application Evaluation Works")
    
    st.markdown("""
    This section explains how the system evaluates an insurance application against the extracted policies and decision trees.
    """)
    
    # Step-by-step process
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        ### Evaluation Steps:
        
        **1️⃣ Policy Matching**
        Identify applicable policies
        
        **2️⃣ Tree Navigation**  
        Ask questions from trees
        
        **3️⃣ Branch Selection**
        Follow answer paths
        
        **4️⃣ Sub-Policy Check**
        Verify child policies
        
        **5️⃣ Final Decision**
        Aggregate all results
        """)
    
    with col2:
        # Example scenario
        with st.expander("💡 **Example: Bariatric Surgery Application**", expanded=True):
            st.markdown("""
            **Applicant Profile:**
            - Name: John Smith, Age: 42
            - BMI: 43 kg/m²
            - Conditions: Type 2 diabetes, hypertension
            - Weight loss attempts: 18 months (failed)
            
            **Evaluation Process:**
            
            **Step 1 - Policy Matching:**
            - ✅ "Medically Necessary Criteria for Bariatric Surgery" (Root Policy)
            - ✅ "Morbid Obesity Criteria" (Related policy)
            - ❌ "Hiatal Hernia Guidelines" (Not applicable - no hernia)
            
            **Step 2 - Decision Tree Navigation:**
            ```
            Q1: What is your current BMI?
            → Answer: 43 kg/m²
            → BMI ≥40? YES ✅ → Continue
            
            Q2: Documented failed medical weight loss?
            → Answer: Yes (18 months)
            → Failed weight loss? YES ✅ → Continue
            
            Q3: Completed multidisciplinary evaluation?
            → Answer: Yes
            → Evaluation complete? YES ✅ → Continue
            
            Q4: Pregnancy considerations?
            → Answer: N/A (male applicant)
            → Not applicable → Continue
            ```
            
            **Step 3 - Sub-Policy Verification:**
            - Check "Bariatric Procedure Considerations": BMI ≥40? ✅ PASS
            - Check "Adolescent Eligibility": Age < 18? ❌ Not applicable (adult)
            
            **Step 4 - Final Decision:**
            ```
            ✅ APPROVED
            
            Conditions Met:
            • BMI ≥40 kg/m² (actual: 43)
            • Failed medical weight loss (18 months documented)
            • Clinically significant comorbidities (diabetes, hypertension)
            • Completed multidisciplinary evaluation
            
            Approved Procedures:
            • Laparoscopic gastric bypass
            • Vertical sleeve gastrectomy
            
            Confidence: 95%
            ```
            """)
        
        # Alternative scenario
        with st.expander("❌ **Counter-Example: Denied Application**", expanded=False):
            st.markdown("""
            **Applicant Profile:**
            - Age: 38, BMI: 38 kg/m²
            - No comorbidities
            - No prior weight loss attempts
            
            **Decision Path:**
            ```
            Q1: Is BMI ≥40?
            → Answer: NO (BMI = 38)
            → Check: BMI 35-39.9 with comorbidity?
               • BMI 35-39.9? YES ✅
               • Comorbidity? NO ❌
            
            ❌ DENIED
            
            Reason: "BMI between 35-40 requires at least one 
            clinically significant obesity-related comorbidity.
            Applicant has no documented comorbidities."
            ```
            """)
    
    st.markdown("---")
    
    # Value proposition
    st.markdown("### 💼 System Benefits:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Before (Manual Process):**
        - ⏱️ 15-30 minutes per application
        - ❓ Inconsistent decisions
        - 📄 Hard to audit
        - 🤷 Difficult to explain reasoning
        """)
    
    with col2:
        st.markdown("""
        **After (Automated System):**
        - ⚡ Seconds per application (95%+ faster)
        - ✅ Consistent every time
        - 📊 Fully auditable trail
        - 💡 Transparent explanations
        """)


# Helper function for editable decision trees
def _display_editable_trees(results_data: dict, job_id: str):
    """Display editable decision trees and policy hierarchy."""

    # Initialize edited data in session state if not present
    if 'edited_hierarchy' not in st.session_state:
        st.session_state.edited_hierarchy = results_data.get("policy_hierarchy", {}).copy() if results_data.get("policy_hierarchy") else {}
    if 'edited_trees' not in st.session_state:
        st.session_state.edited_trees = results_data.get("decision_trees", []).copy() if results_data.get("decision_trees") else []

    # Tabs for different sections
    edit_tab1, edit_tab2, edit_tab3 = st.tabs(["📊 Metadata", "🏛️ Policy Hierarchy", "🌳 Decision Trees"])

    # Tab 1: Metadata (read-only for now)
    with edit_tab1:
        if "metadata" in results_data:
            metadata = results_data["metadata"]
            st.write("**Document Metadata** (Read-only)")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Pages", metadata.get("total_pages", "N/A"))
            with col2:
                st.metric("Document Type", metadata.get("document_type", "N/A"))
            with col3:
                st.metric("Processing Model", "GPT-4" if metadata.get("llm_model", "").endswith("-4") else "GPT-4o-mini")

        if "validation_result" in results_data:
            validation = results_data["validation_result"]
            st.write("**Validation Results** (Read-only)")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall Confidence", f"{validation.get('overall_confidence', 0):.0%}")
            with col2:
                st.metric("Completeness", f"{validation.get('completeness_score', 0):.0%}")
            with col3:
                st.metric("Consistency", f"{validation.get('consistency_score', 0):.0%}")
            with col4:
                st.metric("Traceability", f"{validation.get('traceability_score', 0):.0%}")

    # Tab 2: Policy Hierarchy Editor
    with edit_tab2:
        st.write("**Edit Policy Hierarchy**")
        hierarchy = st.session_state.edited_hierarchy

        if "root_policies" in hierarchy and hierarchy["root_policies"]:
            # Display and edit root policies
            for idx, policy in enumerate(hierarchy["root_policies"]):
                with st.expander(f"📋 Policy {idx + 1}: {policy.get('title', 'Untitled')}", expanded=False):
                    # Editable fields
                    new_title = st.text_input(
                        "Title",
                        value=policy.get('title', ''),
                        key=f"policy_title_{idx}"
                    )
                    new_description = st.text_area(
                        "Description",
                        value=policy.get('description', ''),
                        key=f"policy_desc_{idx}"
                    )

                    # Update if changed
                    if new_title != policy.get('title'):
                        st.session_state.edited_hierarchy["root_policies"][idx]['title'] = new_title
                    if new_description != policy.get('description'):
                        st.session_state.edited_hierarchy["root_policies"][idx]['description'] = new_description

                    # Display conditions
                    if policy.get("conditions"):
                        st.write("**Conditions:**")
                        for cond_idx, cond in enumerate(policy["conditions"]):
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                new_cond = st.text_input(
                                    f"Condition {cond_idx + 1}",
                                    value=cond.get('description', ''),
                                    key=f"cond_{idx}_{cond_idx}"
                                )
                                if new_cond != cond.get('description'):
                                    st.session_state.edited_hierarchy["root_policies"][idx]["conditions"][cond_idx]['description'] = new_cond
                            with col2:
                                if st.button("🗑️", key=f"del_cond_{idx}_{cond_idx}", help="Delete condition"):
                                    st.session_state.edited_hierarchy["root_policies"][idx]["conditions"].pop(cond_idx)
                                    st.rerun()

                        # Add new condition
                        if st.button("➕ Add Condition", key=f"add_cond_{idx}"):
                            if "conditions" not in st.session_state.edited_hierarchy["root_policies"][idx]:
                                st.session_state.edited_hierarchy["root_policies"][idx]["conditions"] = []
                            st.session_state.edited_hierarchy["root_policies"][idx]["conditions"].append({
                                "description": "New condition",
                                "type": "custom"
                            })
                            st.rerun()

                    # Delete policy button
                    if st.button(f"🗑️ Delete Policy {idx + 1}", key=f"del_policy_{idx}", type="secondary"):
                        st.session_state.edited_hierarchy["root_policies"].pop(idx)
                        st.rerun()

            # Add new policy
            st.markdown("---")
            if st.button("➕ Add New Policy", key="add_new_policy"):
                new_policy = {
                    "policy_id": f"new_policy_{len(hierarchy['root_policies']) + 1}",
                    "title": "New Policy",
                    "description": "Add description here",
                    "level": 0,
                    "conditions": [],
                    "children": []
                }
                st.session_state.edited_hierarchy["root_policies"].append(new_policy)
                st.rerun()
        else:
            st.info("No policies found. Add a new policy to get started.")
            if st.button("➕ Add First Policy"):
                st.session_state.edited_hierarchy["root_policies"] = [{
                    "policy_id": "policy_001",
                    "title": "New Policy",
                    "description": "Add description here",
                    "level": 0,
                    "conditions": [],
                    "children": []
                }]
                st.rerun()

    # Tab 3: Decision Trees Editor
    with edit_tab3:
        st.write("**Edit Decision Trees**")
        trees = st.session_state.edited_trees

        if trees:
            for tree_idx, tree in enumerate(trees):
                with st.expander(f"🌳 Tree {tree_idx + 1}: {tree.get('policy_title', 'Untitled')}", expanded=False):
                    # Edit tree title
                    new_tree_title = st.text_input(
                        "Policy Title",
                        value=tree.get('policy_title', ''),
                        key=f"tree_title_{tree_idx}"
                    )
                    if new_tree_title != tree.get('policy_title'):
                        st.session_state.edited_trees[tree_idx]['policy_title'] = new_tree_title

                    # Tree statistics and routing status
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Policy ID", tree.get('policy_id', 'N/A')[:12] + "...")
                    with col2:
                        st.metric("Questions", len(tree.get('questions', [])))
                    with col3:
                        routing_status = "Complete" if tree.get('has_complete_routing', False) else "Incomplete"
                        routing_color = "🟢" if tree.get('has_complete_routing', False) else "🔴"
                        st.metric("Routing", f"{routing_color} {routing_status}")
                    with col4:
                        st.metric("Outcomes", tree.get('total_outcomes', 0))

                    # Show routing issues if any
                    if not tree.get('has_complete_routing', True):
                        with st.container():
                            st.warning("⚠️ Routing Issues Detected")
                            unreachable = tree.get('unreachable_nodes', [])
                            incomplete = tree.get('incomplete_routes', [])

                            if unreachable:
                                st.write(f"**Unreachable Nodes:** {len(unreachable)}")
                                st.caption(", ".join(unreachable[:5]))

                            if incomplete:
                                st.write(f"**Incomplete Routes:** {len(incomplete)}")
                                st.caption(", ".join(incomplete[:5]))

                    st.markdown("---")

                    # Tree visualization
                    with st.container():
                        st.markdown("#### 🌳 Tree Structure Preview")
                        try:
                            render_tile_tree_view(tree)
                        except Exception as e:
                            st.info("Tree structure visualization not available for this tree")
                            logger.debug(f"Error rendering tree: {e}")

                    st.markdown("---")

                    # Edit questions
                    if tree.get("questions"):
                        st.markdown("#### Questions:")
                        for q_idx, question in enumerate(tree["questions"]):
                            st.markdown(f"**Question {q_idx + 1}:**")
                            col1, col2 = st.columns([4, 1])

                            with col1:
                                new_question_text = st.text_area(
                                    "Question Text",
                                    value=question.get('question_text', ''),
                                    key=f"q_text_{tree_idx}_{q_idx}",
                                    height=100
                                )
                                if new_question_text != question.get('question_text'):
                                    st.session_state.edited_trees[tree_idx]["questions"][q_idx]['question_text'] = new_question_text

                                new_answer_type = st.selectbox(
                                    "Answer Type",
                                    options=["boolean", "text", "number", "multiple_choice", "date"],
                                    index=["boolean", "text", "number", "multiple_choice", "date"].index(question.get('answer_type', 'text')),
                                    key=f"q_type_{tree_idx}_{q_idx}"
                                )
                                if new_answer_type != question.get('answer_type'):
                                    st.session_state.edited_trees[tree_idx]["questions"][q_idx]['answer_type'] = new_answer_type

                                # Show routing rules if they exist
                                if question.get('routing_rules'):
                                    with st.expander("🔀 Routing Rules (Read-only)", expanded=False):
                                        for rule_idx, rule in enumerate(question['routing_rules']):
                                            st.caption(f"Rule {rule_idx + 1}:")
                                            st.write(f"  - Answer: {rule.get('answer_value')}")
                                            st.write(f"  - Comparison: {rule.get('comparison')}")
                                            st.write(f"  - Next Node: {rule.get('next_node_id')}")
                                            if rule.get('condition_expression'):
                                                st.write(f"  - Condition: {rule.get('condition_expression')}")

                            with col2:
                                st.write("")  # Spacing
                                st.write("")  # Spacing
                                if st.button("🗑️", key=f"del_q_{tree_idx}_{q_idx}", help="Delete question"):
                                    st.session_state.edited_trees[tree_idx]["questions"].pop(q_idx)
                                    st.rerun()

                            st.markdown("---")

                        # Add new question to tree
                        if st.button(f"➕ Add Question to Tree {tree_idx + 1}", key=f"add_q_{tree_idx}"):
                            new_question = {
                                "question_id": f"q_{len(tree['questions']) + 1}",
                                "question_text": "New question?",
                                "answer_type": "text",
                                "is_required": True
                            }
                            st.session_state.edited_trees[tree_idx]["questions"].append(new_question)
                            st.rerun()

                    # Delete entire tree
                    st.markdown("---")
                    if st.button(f"🗑️ Delete Entire Tree {tree_idx + 1}", key=f"del_tree_{tree_idx}", type="secondary"):
                        st.session_state.edited_trees.pop(tree_idx)
                        st.rerun()

            # Add new decision tree
            st.markdown("---")
            if st.button("➕ Add New Decision Tree", key="add_new_tree"):
                new_tree = {
                    "policy_id": f"new_tree_{len(trees) + 1}",
                    "policy_title": "New Decision Tree",
                    "total_questions": 0,
                    "max_depth": 1,
                    "questions": []
                }
                st.session_state.edited_trees.append(new_tree)
                st.rerun()
        else:
            st.info("No decision trees found. Add a new tree to get started.")
            if st.button("➕ Add First Decision Tree"):
                st.session_state.edited_trees = [{
                    "policy_id": "tree_001",
                    "policy_title": "New Decision Tree",
                    "total_questions": 0,
                    "max_depth": 1,
                    "questions": []
                }]
                st.rerun()


# Helper function for interactive view with branch selector
def _display_interactive_view(results_data: dict):
    """Display interactive policy view with branch selector and tile-based tree visualization."""

    # Branch/Tree Selector
    if "decision_trees" in results_data and results_data["decision_trees"]:
        trees = results_data["decision_trees"]

        st.subheader("🌳 Interactive Decision Tree Explorer")

        # Use tile visualizer's tree selector
        selected_tree_idx = render_tree_selector(trees)

        if selected_tree_idx is not None:
            selected_tree = trees[selected_tree_idx]

            st.markdown("---")

            # Display tile-based tree visualization
            render_tile_tree_view(selected_tree)

            # Show additional tree details
            st.markdown("---")
            st.markdown("### 📋 Additional Tree Information")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Policy ID", selected_tree.get('policy_id', 'N/A'))
            with col2:
                validation_status = "Complete" if selected_tree.get('has_complete_routing', False) else "Incomplete"
                st.metric("Routing Status", validation_status)
            with col3:
                total_outcomes = selected_tree.get('total_outcomes', 0)
                st.metric("Total Outcomes", total_outcomes)

            # Show routing issues if any
            if not selected_tree.get('has_complete_routing', True):
                with st.expander("⚠️ Routing Issues Detected", expanded=True):
                    unreachable = selected_tree.get('unreachable_nodes', [])
                    incomplete = selected_tree.get('incomplete_routes', [])

                    if unreachable:
                        st.warning(f"**Unreachable Nodes:** {len(unreachable)}")
                        st.write(", ".join(unreachable))

                    if incomplete:
                        st.warning(f"**Incomplete Routes:** {len(incomplete)}")
                        st.write(", ".join(incomplete))


# Helper function for displaying decision trees
def _display_decision_trees(results_data: dict):
    """Display decision trees from results data."""

    # Show metadata
    if "metadata" in results_data:
        metadata = results_data["metadata"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Pages", metadata.get("total_pages", "N/A"))
        with col2:
            st.metric("Document Type", metadata.get("document_type", "N/A"))
        with col3:
            st.metric("Processing Model", "GPT-4" if metadata.get("llm_model", "").endswith("-4") else "GPT-4o-mini")

    st.markdown("---")

    # Show validation results
    if "validation_result" in results_data:
        validation = results_data["validation_result"]

        st.subheader("Validation Results")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Overall Confidence",
                f"{validation.get('overall_confidence', 0):.0%}",
                delta="Pass" if validation.get('is_valid') else "Fail"
            )
        with col2:
            st.metric("Completeness", f"{validation.get('completeness_score', 0):.0%}")
        with col3:
            st.metric("Consistency", f"{validation.get('consistency_score', 0):.0%}")
        with col4:
            st.metric("Traceability", f"{validation.get('traceability_score', 0):.0%}")

    st.markdown("---")

    # Show policy hierarchy with visualization
    if "policy_hierarchy" in results_data:
        hierarchy = results_data["policy_hierarchy"]
        display_policy_hierarchy(hierarchy)

    st.markdown("---")
    
    # Add evaluation guide
    _display_evaluation_flow_guide()
    
    st.markdown("---")

    # Show decision trees with visualization
    if "decision_trees" in results_data:
        trees = results_data["decision_trees"]

        st.markdown(f"## 🌳 Decision Trees ({len(trees)} Generated)")
        
        st.info("💡 **Tip:** Use the tabs below to view questions in different formats and explore how decisions are made")
        
        for i, tree in enumerate(trees, 1):
            _display_decision_tree_interactive(tree, i)
            if i < len(trees):
                st.markdown("---")

    # Export button
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        # Export JSON
        json_str = json.dumps(results_data, indent=2, default=str)
        st.download_button(
            "Export as JSON",
            data=json_str,
            file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="export_json_view"
        )

    with col2:
        # Export Decision Trees only
        if "decision_trees" in results_data:
            trees_json = json.dumps(results_data["decision_trees"], indent=2, default=str)
            st.download_button(
                "Export Decision Trees",
                data=trees_json,
                file_name=f"decision_trees_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="export_trees_view"
            )

# Main content - Tabs
# Main content - Simplified Tabs (2 tabs instead of 3)
tab1, tab2 = st.tabs(["Upload Document", "Policies"])

# ============================================================================
# TAB 1: Upload & Process
# ============================================================================
with tab1:
    st.header("Upload Policy Document")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Policy name input
        policy_name = st.text_input(
            "Policy Name *",
            placeholder="Enter a unique name for this policy (e.g., 'Health Insurance 2024')",
            help="Provide a unique, descriptive name for this policy document"
        )

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a PDF policy document",
            type=["pdf"],
            help="Upload an insurance policy, legal document, or corporate policy PDF"
        )

    with col2:
        st.subheader("Processing Options")
        use_gpt4 = st.checkbox(
            "Use GPT-4",
            value=False,
            help="Use GPT-4 for complex documents (higher cost, better accuracy)"
        )

        enable_streaming = st.checkbox(
            "Enable Streaming",
            value=True,
            help="Stream processing progress in real-time"
        )

        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum confidence for validation"
        )

    # Process button
    if uploaded_file is not None:
        st.markdown("---")

        # Show file info
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.info(f"**File:** {uploaded_file.name} | **Size:** {file_size_mb:.2f} MB")

        # Validation and processing
        if st.button(" Process Document", type="primary", use_container_width=True):
            # Validate policy name
            if not policy_name or not policy_name.strip():
                st.error("⚠️ Please provide a policy name before processing.")
            elif db_ops.policy_name_exists(policy_name.strip()):
                st.error(f"⚠️ Policy name '{policy_name.strip()}' already exists. Please choose a unique name.")
            else:
                with st.spinner("Processing document... This may take a few minutes."):
                    try:
                        # Process the document
                        result = a2a_client.process_document(
                            pdf_bytes=uploaded_file.getvalue(),
                            filename=uploaded_file.name,
                            use_gpt4=use_gpt4,
                            enable_streaming=enable_streaming,
                            confidence_threshold=confidence_threshold
                        )

                        # Check for errors
                        if result.get("status") == "failed":
                            st.error("Document processing failed!")
                            st.error(result.get("message", "Unknown error"))
                        else:
                            # Save results to database using backend handler
                            saved_result = backend_handler.process_a2a_response(
                                response=result,
                                pdf_bytes=uploaded_file.getvalue(),
                                filename=uploaded_file.name,
                                policy_name=policy_name.strip(),
                                use_gpt4=use_gpt4,
                                confidence_threshold=confidence_threshold
                            )

                            if saved_result.get("saved_to_database"):
                                st.success(f"✅ Policy '{policy_name.strip()}' processed and saved successfully!")

                                job_id = saved_result.get("job_id")
                                if job_id:
                                    # Store in session state for tab 2
                                    st.session_state.last_job_id = job_id

                                    # Display summary
                                    results_data = saved_result.get("results", {})

                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Policy Name", policy_name.strip())
                                    with col2:
                                        total_policies = results_data.get("policy_hierarchy", {}).get("total_policies", 0)
                                        st.metric("Total Policies", total_policies)
                                    with col3:
                                        total_trees = len(results_data.get("decision_trees", []))
                                        st.metric("Decision Trees", total_trees)

                                    st.info(f"Job ID: `{job_id}` - Switch to 'Review Decision Trees' tab to view results")
                            else:
                                st.warning("Document processed but failed to save to database")
                                st.json(result)

                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
                        logger.error(f"Processing error: {e}", exc_info=True)

    # Recent Jobs Section
    st.markdown("---")
    st.subheader("Recent Processing Jobs")

    # Get recent jobs
    recent_jobs = db_ops.list_jobs(limit=10)

    if recent_jobs:
        for job in recent_jobs:
            policy_name_display = job.get('policy_name', 'Unnamed Policy')
            with st.expander(
                f"{policy_name_display} - {job['status'].upper()} - {job.get('created_at', 'Unknown time')}"
            ):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write(f"**Policy Name:** {policy_name_display}")
                    st.write(f"**Status:** {job['status']}")
                    st.write(f"**Type:** {job.get('document_type', 'N/A')}")

                with col2:
                    st.write(f"**Job ID:** {job['job_id'][:12]}...")
                    st.write(f"**Policies:** {job.get('total_policies', 'N/A')}")
                    st.write(f"**Confidence:** {job.get('validation_confidence', 0):.0%}" if job.get('validation_confidence') else "**Confidence:** N/A")

                with col3:
                    st.write(f"**Processing Time:** {job.get('processing_time_seconds', 'N/A')}s" if job.get('processing_time_seconds') else "**Processing Time:** N/A")

                # Load results button
                if job['status'] == 'completed':
                    if st.button(f"View Results", key=f"view_{job['job_id']}"):
                        st.session_state.selected_job_id = job['job_id']
                        st.rerun()
    else:
        st.info("No jobs found. Upload a document to get started!")

# ============================================================================
# TAB 2: Policies (Simplified - View & Edit Combined)
# ============================================================================
with tab2:
    st.header("Policies")

    # Get all completed policies
    all_policies = db_ops.get_all_policies(status="completed")

    if not all_policies:
        st.info("No policies found. Upload and process a document to get started.")
    else:
        # Simple policy selector
        policy_options = {f"{p['policy_name']}": p['job_id'] for p in all_policies}
        policy_names = list(policy_options.keys())

        selected_policy_name = st.selectbox(
            "Select a Policy",
            options=policy_names,
            help="Choose a policy to view"
        )

        selected_job_id = policy_options[selected_policy_name]

        # Load button
        if st.button("Load Policy", type="primary"):
            with st.spinner(f"Loading '{selected_policy_name}'..."):
                try:
                    results = backend_handler.get_job_results(selected_job_id)
                    if results:
                        st.session_state.current_results = results
                        st.session_state.current_job_id = selected_job_id
                        st.session_state.current_policy_name = selected_policy_name
                        st.success("✅ Policy loaded!")
                    else:
                        st.error("No results found for this policy")
                except Exception as e:
                    st.error(f"Error loading policy: {str(e)}")

        # Display loaded policy with side-by-side PDF and tree view
        if 'current_results' in st.session_state and st.session_state.current_results:
            st.markdown("---")
            
            results_data = st.session_state.current_results
            
            # Display decision trees with side-by-side layout
            if "decision_trees" in results_data:
                trees = results_data["decision_trees"]
                
                if len(trees) == 0:
                    st.warning("No decision trees found in this policy")
                else:
                    # Header with policy name
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 20px; border-radius: 12px; margin-bottom: 20px;'>
                        <h2 style='color: white; margin: 0;'>📋 {st.session_state.current_policy_name}</h2>
                        <p style='color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-size: 0.95em;'>
                            {len(trees)} Decision Tree Branch{"es" if len(trees) > 1 else ""} Available
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create branch selector dropdown
                    st.markdown("### 🌳 Select Decision Tree Branch")
                    
                    # Build dropdown options
                    tree_options = {}
                    for i, tree in enumerate(trees):
                        policy_title = tree.get('policy_title', f'Tree {i+1}')
                        policy_id = tree.get('policy_id', 'N/A')
                        tree_options[f"{policy_title}"] = i
                    
                    # Branch selector
                    selected_tree_name = st.selectbox(
                        "Select a Decision Tree Branch to View:",
                        options=list(tree_options.keys()),
                        help="Choose which decision tree branch you want to explore"
                    )
                    
                    # Get selected tree index
                    selected_tree_idx = tree_options[selected_tree_name]
                    selected_tree = trees[selected_tree_idx]
                    
                    st.markdown("---")
                    
                    # Create side-by-side layout: PDF viewer | Decision Tree
                    col_left, col_right = st.columns([1, 1], gap="large")
                    
                    with col_left:
                        st.markdown("### 📄 Source Document")
                        
                        # Check if we have the original PDF
                        if 'document_content' in results_data and results_data['document_content']:
                            import base64
                            
                            # Display PDF
                            pdf_data = results_data['document_content']
                            
                            # Create PDF viewer
                            pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
                            pdf_display = f'''
                            <iframe 
                                src="data:application/pdf;base64,{pdf_base64}" 
                                width="100%" 
                                height="800" 
                                type="application/pdf"
                                style="border: 2px solid #e0e0e0; border-radius: 8px;">
                            </iframe>
                            '''
                            st.markdown(pdf_display, unsafe_allow_html=True)
                        else:
                            st.info("📄 PDF preview not available for this policy")
                            
                            # Show metadata instead
                            st.markdown("#### 📊 Policy Metadata")
                            if 'metadata' in results_data:
                                metadata = results_data['metadata']
                                st.write(f"**Pages:** {metadata.get('total_pages', 'N/A')}")
                                st.write(f"**Processing Time:** {metadata.get('processing_time', 'N/A')}s")
                                st.write(f"**Confidence:** {metadata.get('validation_confidence', 0):.0%}")
                    
                    with col_right:
                        st.markdown("### 🌳 Decision Tree Structure")
                        
                        # Display the selected tree with visualization
                        display_decision_tree(selected_tree, selected_tree_idx + 1)
            else:
                st.warning("No decision trees found in this policy")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Policy Document Processor"
    "</div>",
    unsafe_allow_html=True
)
