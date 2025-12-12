"""
Decision Tree Renderer

This module provides visualization for decision trees with:
- Hierarchical depth indicators
- Clear parent-child relationships
- IF-THEN answer path visualization
- Color-coded outcomes (approved, denied, review)
- Debug capabilities for data inspection
"""

import streamlit as st
from typing import Dict, List, Any, Optional


def display_decision_tree(tree: dict, tree_idx: int):
    """
    Display decision tree with depth visualization and clear indentation.
    
    Args:
        tree: Dictionary containing tree data with keys:
            - policy_title: Title of the policy
            - policy_id: Unique identifier for the policy
            - questions: List of question dictionaries
            - root_node or tree.root: Root node of the decision tree
        tree_idx: Index of the tree (for display purposes)
    """
    
    policy_title = tree.get('policy_title', f'Decision Tree {tree_idx}')
    policy_id = tree.get('policy_id', 'N/A')
    
    # Debug: Show tree structure
    with st.expander("[DEBUG] Debug: Tree Data Structure", expanded=False):
        st.write("**Tree Keys:**", list(tree.keys()))
        st.write("**Policy Title:**", policy_title)
        st.write("**Policy ID:**", policy_id)
        if 'questions' in tree:
            st.write("**Questions Count:**", len(tree['questions']))
            if tree['questions']:
                st.write("**First Question:**", tree['questions'][0])
        if 'root_node' in tree:
            st.write("**Root Node Keys:**", list(tree['root_node'].keys()) if isinstance(tree['root_node'], dict) else type(tree['root_node']))
            st.write("**Root Node:**", tree['root_node'])
    
    # Tree header with metadata
    st.markdown(f"""
    <div style='padding: 20px; 
                background: linear-gradient(135deg, #4A5ED7 0%, #5B4DB2 100%); 
                border-radius: 10px; margin-bottom: 20px; color: white; 
                box-shadow: 0 3px 6px rgba(0,0,0,0.15);'>
        <h3 style='margin: 0; color: white; font-weight: 600;'>{policy_title}</h3>
        <p style='margin: 8px 0 0 0; color: rgba(255,255,255,0.9); font-size: 0.9em;'>
            <strong>Policy ID:</strong> <code style='background: rgba(255,255,255,0.2); 
            padding: 2px 8px; border-radius: 4px;'>{policy_id}</code>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get questions and root node
    questions = tree.get("questions", [])
    tree_root = tree.get('root_node', tree.get('tree', {}).get('root', {}))
    
    if not questions and not tree_root:
        st.warning("No questions or tree structure found")
        return
    
    # Display hierarchical tree structure with depth indicators
    render_tree_with_depth(tree_root, questions, tree_idx)


def render_tree_with_depth(root_node: dict, questions: list, tree_idx: int):
    """
    Render decision tree with clear depth visualization and hierarchical indentation.
    
    Args:
        root_node: Root node of the decision tree
        questions: List of question dictionaries
        tree_idx: Index of the tree
    """
    
    if not root_node and not questions:
        st.info("No decision tree structure available")
        return
    
    # Display questions with hierarchical depth
    if questions:
        st.markdown("""
        <div style='background: #F8F9FA; padding: 15px; border-radius: 8px; margin-bottom: 20px;'>
            <h4 style='margin: 0 0 10px 0; color: #1F2937;'>[FLOW] Decision Flow Path</h4>
            <p style='margin: 0; color: #6B7280; font-size: 0.9em;'>
                Follow the questions below to determine the outcome. Each question leads to specific paths based on your answers.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create question lookup map
        question_map = {q.get('question_id'): q for q in questions}
        
        # Render from root node if available
        if root_node:
            render_node_recursive(root_node, question_map, 0, tree_idx, path_num=1)
        else:
            # Fallback: render questions linearly
            for q_idx, question in enumerate(questions, 1):
                render_question_tile(question, q_idx, 0, tree_idx)
    
    # Display conditions if available
    elif root_node and 'conditions' in root_node and root_node['conditions']:
        st.markdown("---")
        render_conditions_as_tiles(root_node['conditions'], 0)


def render_node_recursive(node: dict, question_map: dict, depth: int, tree_idx: int, path_num: int = 1):
    """
    Recursively render decision tree nodes with proper hierarchy.
    
    Args:
        node: Current node in the tree
        question_map: Dictionary mapping question IDs to question data
        depth: Current depth level in the tree
        tree_idx: Index of the tree
        path_num: Current path number for display
    """
    
    if not node:
        return
    
    node_type = node.get('node_type', node.get('type', 'unknown'))
    node_id = node.get('node_id', node.get('id', 'unknown'))
    
    # Question node
    if node_type == 'question':
        # Get question from node itself or from question map
        question = node.get('question')
        if not question:
            question_id = node.get('question_id', node_id)
            question = question_map.get(question_id)
        
        if question:
            render_question_tile(question, path_num, depth, tree_idx)
            
            # Render child paths - handle both list and dict formats
            children = node.get('children', node.get('branches', {}))
            
            # If children is a dict (answer_key -> node), convert to branches format
            if isinstance(children, dict):
                for idx, (answer_key, child_node) in enumerate(children.items(), 1):
                    child_node_type = child_node.get('node_type', 'unknown')
                    
                    # Determine if this is a terminal outcome or continuation
                    is_terminal = child_node_type == 'outcome'
                    
                    if is_terminal:
                        # Create a branch representation for terminal outcome
                        branch = {
                            'answer': answer_key,
                            'child_node': child_node,
                            'is_terminal': True,
                            'outcome_type': child_node.get('outcome_type', ''),
                            'outcome_label': child_node.get('outcome', ''),
                            'explanation': child_node.get('outcome', '')
                        }
                        render_branch_with_outcome(branch, question, tree_idx, depth + 1, idx, path_num)
                    else:
                        # For continuation to next question, show a simpler connector
                        render_answer_connector(answer_key, depth + 1, idx)
                        
                        # Then recursively render the child question node
                        render_node_recursive(child_node, question_map, depth + 1, tree_idx, path_num + idx)
            else:
                # Handle list format
                for idx, branch in enumerate(children, 1):
                    render_branch_with_outcome(branch, question, tree_idx, depth + 1, idx, path_num)
    
    # Outcome node
    elif node_type == 'outcome':
        outcome_data = {
            'outcome_type': node.get('outcome_type', 'unknown'),
            'outcome_label': node.get('outcome', node.get('label', 'Decision')),
            'description': node.get('outcome', node.get('explanation', node.get('description', '')))
        }
        render_outcome_tile(outcome_data, tree_idx, depth, path_num)


def render_answer_connector(answer: str, depth: int, branch_num: int):
    """
    Render a simple connector showing which answer path leads to the next question.
    
    Args:
        answer: The answer value (e.g., "yes", "no", ">=40")
        depth: Indentation depth level
        branch_num: Branch number for display
    """
    indent = depth * 30 + 40
    
    st.markdown(f"""
    <div style='margin-left: {indent}px; margin-bottom: 8px;'>
        <div style='border-left: 3px solid #94A3B8; padding-left: 12px;'>
            <div style='display: inline-block; background: rgba(249,115,22,0.1); 
                        padding: 4px 10px; border-radius: 6px; border: 1px solid #F97316;'>
                <span style='color: #EA580C; font-size: 0.8em; font-weight: 600;'>IF:</span>
                <code style='background: rgba(249,115,22,0.15); padding: 2px 8px; 
                             border-radius: 4px; color: #C2410C; margin-left: 6px; font-weight: 600;'>{answer}</code>
                <span style='color: #F97316; margin-left: 6px;'>--> THEN</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_question_tile(question: dict, step_num: int, depth: int, tree_idx: int):
    """
    Render a single question tile with proper styling and indentation.
    
    Args:
        question: Question dictionary containing:
            - question_text: The question text
            - question_type: Type of answer expected
            - question_id: Unique identifier
            - explanation: Optional explanation
        step_num: Step number for display
        depth: Indentation depth level
        tree_idx: Index of the tree
    """
    
    question_text = question.get('question_text', question.get('question', question.get('text', 'N/A')))
    answer_type = question.get('question_type', question.get('answer_type', question.get('type', 'unknown')))
    question_id = question.get('question_id', f'q{step_num}')
    explanation = question.get('explanation', '')
    
    indent = depth * 30
    
    st.markdown(f"""
    <div style='margin-left: {indent}px; margin-bottom: 20px;'>
        <div style='padding: 20px; 
                    background-color: #1E3A8A; 
                    border-left: 6px solid #3B82F6; 
                    border-radius: 10px; 
                    box-shadow: 0 2px 6px rgba(0,0,0,0.1);'>
            <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 12px;'>
                <span style='background: #3B82F6; color: white; padding: 6px 14px; 
                             border-radius: 20px; font-weight: 600; font-size: 0.85em;'>
                    STEP {step_num}
                </span>
                <span style='background: rgba(59,130,246,0.2); color: #93C5FD; padding: 4px 10px; 
                             border-radius: 12px; font-size: 0.8em;'>
                    {question_id}
                </span>
            </div>
            <h4 style='margin: 0 0 12px 0; color: #FFFFFF; font-weight: 500; font-size: 1.1em;'>
                [?] {question_text}
            </h4>
            <div style='color: #93C5FD; font-size: 0.9em; margin-bottom: 8px;'>
                <strong>Answer Type:</strong> 
                <code style='background: rgba(255,255,255,0.1); padding: 3px 10px; 
                             border-radius: 4px; color: #DBEAFE; margin-left: 8px;'>{answer_type}</code>
            </div>
            {f"<div style='color: #D1FAE5; font-size: 0.85em; margin-top: 8px; font-style: italic;'>[INFO] {explanation}</div>" if explanation else ""}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_branch_with_outcome(branch: dict, question: dict, tree_idx: int, depth: int, branch_num: int, question_num: int):
    """
    Render a terminal outcome branch from a question.
    This function now only handles terminal outcomes (approved/denied/review).
    For continuation to next question, use render_answer_connector instead.
    
    Args:
        branch: Branch dictionary containing answer and outcome
        question: Parent question dictionary
        tree_idx: Index of the tree
        depth: Indentation depth level
        branch_num: Branch number for display
        question_num: Question number for display
    """
    
    # Extract branch data - handle both old and new formats
    answer = branch.get('answer', branch.get('value', branch.get('condition', 'N/A')))
    
    # Check if this branch has a child_node (new format)
    child_node = branch.get('child_node', {})
    if child_node:
        # New format: child_node contains the full node
        outcome_type = child_node.get('outcome_type', 'unknown')
        outcome_label = child_node.get('outcome', 'Decision')
        outcome_desc = child_node.get('outcome', '')
    else:
        # Old format: data directly in branch
        outcome = branch.get('outcome', {})
        outcome_type = branch.get('outcome_type', outcome.get('type', 'unknown') if isinstance(outcome, dict) else 'unknown')
        outcome_label = branch.get('outcome_label', outcome.get('label', 'Decision') if isinstance(outcome, dict) else str(outcome))
        outcome_desc = branch.get('explanation', outcome.get('description', '') if isinstance(outcome, dict) else '')
    
    indent = depth * 30 + 40
    
    # Color coding based on outcome type
    if 'approv' in outcome_type.lower() or 'approv' in outcome_label.lower():
        bg_color = '#065F46'; border_color = '#10B981'; icon = '[APPROVED]'; badge = 'APPROVED'
    elif 'den' in outcome_type.lower() or 'den' in outcome_label.lower():
        bg_color = '#991B1B'; border_color = '#EF4444'; icon = '[DENIED]'; badge = 'DENIED'
    elif 'review' in outcome_type.lower() or 'refer' in outcome_type.lower() or 'requires' in outcome_type.lower():
        bg_color = '#92400E'; border_color = '#F59E0B'; icon = '[REVIEW]'; badge = 'REVIEW'
    else:
        bg_color = '#6B21A8'; border_color = '#A855F7'; icon = '[OUTCOME]'; badge = 'OUTCOME'
    
    st.markdown(f"""
    <div style='margin-left: {indent}px; margin-bottom: 15px;'>
        <div style='border-left: 3px solid #94A3B8; padding-left: 12px;'>
            <div style='padding: 16px; background-color: {bg_color}; 
                        border-left: 5px solid {border_color}; border-radius: 10px; 
                        box-shadow: 0 2px 6px rgba(0,0,0,0.15);'>
                <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 10px;'>
                    <span style='background: {border_color}; color: #1F2937; padding: 3px 10px; 
                                 border-radius: 14px; font-weight: 700; font-size: 0.7em;'>
                        {badge}
                    </span>
                </div>
                <div style='margin-bottom: 10px;'>
                    <div style='display: inline-block; background: rgba(255,255,255,0.15); 
                                padding: 4px 10px; border-radius: 5px;'>
                        <span style='color: #E5E7EB; font-size: 0.8em; font-weight: 600;'>IF:</span>
                        <code style='background: rgba(0,0,0,0.2); padding: 3px 8px; 
                                     border-radius: 4px; color: #FCD34D; margin-left: 6px;'>{answer}</code>
                        <span style='color: #E5E7EB; margin-left: 6px; font-size: 0.8em; font-weight: 600;'>--> THEN</span>
                    </div>
                </div>
                <div style='display: flex; align-items: flex-start; gap: 10px; margin-top: 10px;'>
                    <span style='font-size: 1.5em; line-height: 1;'>{icon}</span>
                    <div style='flex: 1;'>
                        <div style='font-size: 1em; color: #FFFFFF; font-weight: 600; line-height: 1.3;'>
                            {outcome_label}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_outcome_tile(outcome_data: dict, tree_idx: int, depth: int, path_num: int):
    """
    Render a terminal outcome tile.
    
    Args:
        outcome_data: Outcome dictionary containing:
            - outcome_type: Type of outcome (approved, denied, review, etc.)
            - outcome_label: Display label for the outcome
            - description: Optional description
        tree_idx: Index of the tree
        depth: Indentation depth level
        path_num: Path number for display
    """
    
    outcome_type = outcome_data.get('outcome_type', 'unknown')
    outcome_label = outcome_data.get('outcome_label', 'Decision')
    outcome_desc = outcome_data.get('description', '')
    
    indent = depth * 30
    
    # Color coding
    if 'approv' in outcome_type.lower() or 'approv' in outcome_label.lower():
        bg_color = '#065F46'; border_color = '#10B981'; icon = '[APPROVED]'; badge = 'APPROVED'
    elif 'den' in outcome_type.lower() or 'den' in outcome_label.lower():
        bg_color = '#991B1B'; border_color = '#EF4444'; icon = '[DENIED]'; badge = 'DENIED'
    elif 'review' in outcome_type.lower() or 'refer' in outcome_type.lower():
        bg_color = '#92400E'; border_color = '#F59E0B'; icon = '[REVIEW]'; badge = 'REVIEW'
    else:
        bg_color = '#6B21A8'; border_color = '#A855F7'; icon = '[OUTCOME]'; badge = 'OUTCOME'
    
    st.markdown(f"""
    <div style='margin-left: {indent}px; margin-bottom: 15px;'>
        <div style='padding: 18px; background-color: {bg_color}; 
                    border-left: 6px solid {border_color}; border-radius: 10px; 
                    box-shadow: 0 2px 6px rgba(0,0,0,0.15);'>
            <div style='display: flex; align-items: center; gap: 12px;'>
                <span style='font-size: 1.8em;'>{icon}</span>
                <div style='flex: 1;'>
                    <div style='background: {border_color}; color: #1F2937; padding: 4px 12px; 
                                 border-radius: 16px; font-weight: 700; font-size: 0.75em; 
                                 display: inline-block; margin-bottom: 8px;'>
                        {badge}
                    </div>
                    <div style='font-size: 1.2em; color: #FFFFFF; font-weight: 600;'>
                        {outcome_label}
                    </div>
                    {f"<div style='color: #D1D5DB; font-size: 0.9em; margin-top: 6px; line-height: 1.4;'>{outcome_desc}</div>" if outcome_desc else ""}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_conditions_as_tiles(conditions: list, depth: int):
    """
    Render a list of conditions as styled tiles.
    
    Args:
        conditions: List of condition dictionaries
        depth: Indentation depth level
    """
    indent = depth * 30
    
    for idx, condition in enumerate(conditions, 1):
        condition_text = condition.get('description', condition.get('text', 'No description'))
        
        st.markdown(f"""
        <div style='margin-left: {indent}px; margin-bottom: 12px;'>
            <div style='padding: 15px; background-color: #F3F4F6; 
                        border-left: 4px solid #6366F1; border-radius: 8px;'>
                <div style='color: #1F2937; font-weight: 500;'>
                    <span style='background: #6366F1; color: white; padding: 2px 8px; 
                                 border-radius: 12px; font-size: 0.75em; margin-right: 10px;'>
                        {idx}
                    </span>
                    {condition_text}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
