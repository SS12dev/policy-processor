"""
Tile-based vertical decision tree visualizer.

Creates an intuitive vertical tree layout with:
- Indented tiles showing hierarchy
- Left-side connection lines
- Color-coded node types
- Answer labels on paths
- Collapsible branches
"""
import streamlit as st
from typing import Dict, List, Optional, Set
from models.schemas import DecisionNode, DecisionTree


class TileTreeVisualizer:
    """Renders decision trees as vertical tiles with indentation."""

    def __init__(self):
        self.node_colors = {
            "question": "#3498db",      # Blue
            "decision": "#f39c12",      # Orange
            "outcome": {
                "approved": "#27ae60",   # Green
                "denied": "#e74c3c",     # Red
                "refer_to_manual": "#e67e22",  # Dark Orange
                "pending_review": "#9b59b6",    # Purple
                "requires_documentation": "#f1c40f"  # Yellow
            }
        }

    def render_tree(self, tree: DecisionTree):
        """
        Render complete decision tree as vertical tiles.

        Args:
            tree: DecisionTree to visualize
        """
        st.markdown(f"### {tree.policy_title}")

        if tree.total_nodes == 0:
            st.warning("Tree has no nodes")
            return

        # Tree statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Nodes", tree.total_nodes)
        with col2:
            st.metric("Total Paths", tree.total_paths)
        with col3:
            st.metric("Max Depth", tree.max_depth)
        with col4:
            score_color = "green" if tree.confidence_score > 0.8 else "orange" if tree.confidence_score > 0.6 else "red"
            st.metric("Confidence", f"{tree.confidence_score:.0%}")

        # Validation status
        if hasattr(tree, 'has_complete_routing'):
            if tree.has_complete_routing:
                st.success("All paths complete")
            else:
                st.error(f"Incomplete routing: {len(tree.incomplete_routes)} nodes")

        st.markdown("---")

        # Render tree nodes
        if 'expanded_nodes' not in st.session_state:
            st.session_state.expanded_nodes = {tree.root_node.node_id}

        self._render_node(tree.root_node, level=0, parent_answer=None)

    def _render_node(
        self,
        node: DecisionNode,
        level: int,
        parent_answer: Optional[str] = None
    ):
        """
        Render a single node with its children.

        Args:
            node: DecisionNode to render
            level: Indentation level
            parent_answer: Answer that led to this node
        """
        # Calculate indentation
        indent = level * 40  # 40px per level

        # Determine color based on node type
        if node.node_type == "outcome":
            color = self.node_colors["outcome"].get(
                node.outcome_type,
                self.node_colors["outcome"]["refer_to_manual"]
            )
        else:
            color = self.node_colors.get(node.node_type, "#95a5a6")

        # Build node HTML
        node_html = self._build_node_html(node, color, indent, parent_answer)

        # Render node using HTML container
        # Use st.html() if available (Streamlit >= 1.31), otherwise use markdown
        try:
            st.html(node_html)
        except AttributeError:
            # Fallback to markdown for older Streamlit versions
            st.markdown(node_html, unsafe_allow_html=True)

        # Handle expansion
        is_expanded = node.node_id in st.session_state.expanded_nodes

        # Toggle button for non-outcome nodes with children
        if node.children and node.node_type != "outcome":
            col1, col2 = st.columns([1, 11])
            with col1:
                expand_label = "[-]" if is_expanded else "[+]"
                if st.button(expand_label, key=f"toggle_{node.node_id}"):
                    if is_expanded:
                        st.session_state.expanded_nodes.remove(node.node_id)
                    else:
                        st.session_state.expanded_nodes.add(node.node_id)
                    st.rerun()

        # Render children if expanded
        if is_expanded and node.children:
            for answer, child_node in node.children.items():
                self._render_node(child_node, level + 1, parent_answer=answer)

    def _build_node_html(
        self,
        node: DecisionNode,
        color: str,
        indent: int,
        parent_answer: Optional[str]
    ) -> str:
        """Build HTML for a single node tile."""

        # Convert hex color to rgba for background (with alpha transparency)
        def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
            """Convert hex color to rgba format."""
            hex_color = hex_color.lstrip('#')
            if len(hex_color) == 6:
                r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
                return f"rgba({r}, {g}, {b}, {alpha})"
            return hex_color

        bg_color = hex_to_rgba(color)

        # Connection line styling
        line_style = f"border-left: 3px solid {color}; margin-left: {indent}px; padding-left: 15px;"

        # Node content
        if node.node_type == "question":
            icon = "[?]"
            title = node.question.question_text if node.question else "No question"
            subtitle = f"Type: {node.question.question_type.value}" if node.question else ""
        elif node.node_type == "decision":
            icon = "[D]"
            title = node.decision_logic or "Logic check"
            subtitle = f"Decision node"
        elif node.node_type == "outcome":
            icon = {"approved": "[APPROVED]", "denied": "[DENIED]", "refer_to_manual": "[MANUAL]"}.get(
                node.outcome_type, "[OUTCOME]"
            )
            title = node.outcome or "Outcome"
            subtitle = f"Outcome: {node.outcome_type}"
        else:
            icon = "[NODE]"
            title = f"Node {node.node_id}"
            subtitle = node.node_type

        # Answer label if from parent
        answer_label = f"<div style='color: {color}; font-weight: bold; margin-bottom: 5px;'>[{parent_answer.upper()}] --></div>" if parent_answer else ""

        html = f"""<div style="{line_style} margin-bottom: 10px;">
{answer_label}
<div style="background-color: {bg_color}; border-left: 4px solid {color}; padding: 10px; border-radius: 5px;">
<div style="font-size: 1.1em; font-weight: bold; color: {color};">{icon} {title}</div>
<div style="font-size: 0.9em; color: #666; margin-top: 5px;">{subtitle}</div>
{self._build_node_details(node)}
</div>
</div>"""

        return html

    def _build_node_details(self, node: DecisionNode) -> str:
        """Build additional details for node."""
        details = []

        # Confidence score
        if node.confidence_score:
            conf_color = "#27ae60" if node.confidence_score > 0.8 else "#f39c12" if node.confidence_score > 0.6 else "#e74c3c"
            details.append(f"<span style='color: {conf_color};'>Confidence: {node.confidence_score:.0%}</span>")

        # Source references
        if node.source_references:
            ref = node.source_references[0]
            details.append(f"<span style='color: #7f8c8d;'>Source: Page {ref.page_number}</span>")

        # Routing info
        if node.routing_rules:
            details.append(f"<span style='color: #3498db;'>Routes: {len(node.routing_rules)}</span>")

        # Child count
        if node.children:
            details.append(f"<span style='color: #9b59b6;'>Branches: {len(node.children)}</span>")

        if not details:
            return ""

        return f"<div style='font-size: 0.8em; margin-top: 8px; color: #7f8c8d;'>{' | '.join(details)}</div>"


def render_tree_selector(trees: List[Dict]) -> Optional[int]:
    """
    Render dropdown to select decision tree branch.

    Args:
        trees: List of tree dictionaries

    Returns:
        Selected tree index or None
    """
    if not trees:
        st.warning("No decision trees available")
        return None

    tree_options = {
        f"{tree.get('policy_title', f'Tree {i+1}')} ({len(tree.get('questions', []))} questions)": i
        for i, tree in enumerate(trees)
    }

    selected_label = st.selectbox(
        "Select Decision Tree Branch",
        options=list(tree_options.keys()),
        help="Choose a branch to visualize"
    )

    return tree_options[selected_label] if selected_label else None


def render_tile_tree_view(tree_data: Dict):
    """
    Render tile-based tree visualization.

    Args:
        tree_data: Tree data dictionary from database
    """
    st.subheader("Decision Tree Visualization")

    # Convert dict to DecisionTree object if needed
    if isinstance(tree_data, dict):
        try:
            from models.schemas import DecisionTree
            tree = DecisionTree(**tree_data)
        except Exception as e:
            st.error(f"Error loading tree: {e}")
            return
    else:
        tree = tree_data

    # Render with visualizer
    visualizer = TileTreeVisualizer()
    visualizer.render_tree(tree)


def render_tree_comparison(tree1: DecisionTree, tree2: DecisionTree):
    """
    Render side-by-side comparison of two trees.

    Args:
        tree1: First decision tree
        tree2: Second decision tree
    """
    col1, col2 = st.columns(2)

    visualizer = TileTreeVisualizer()

    with col1:
        st.markdown("### Tree 1")
        visualizer.render_tree(tree1)

    with col2:
        st.markdown("### Tree 2")
        visualizer.render_tree(tree2)


def render_tree_path_highlighter(tree: DecisionTree, answers: Dict[str, str]):
    """
    Highlight a specific path through the tree based on answers.

    Args:
        tree: DecisionTree to visualize
        answers: Dict mapping question_id to answer value
    """
    st.markdown("### Path Through Tree")
    st.info(f"Following path with {len(answers)} answers")

    # Trace path through tree
    current_node = tree.root_node
    path = [current_node.node_id]

    while current_node.node_type != "outcome":
        # Get answer for this question
        if current_node.question:
            answer = answers.get(current_node.question.question_id)
            if answer and answer in current_node.children:
                current_node = current_node.children[answer]
                path.append(current_node.node_id)
            else:
                st.warning(f"No answer provided for: {current_node.question.question_text}")
                break
        else:
            break

    # Highlight path
    st.session_state.highlighted_path = set(path)

    visualizer = TileTreeVisualizer()
    visualizer.render_tree(tree)

    # Show outcome
    if current_node.node_type == "outcome":
        outcome_color = {
            "approved": "success",
            "denied": "error",
            "refer_to_manual": "warning"
        }.get(current_node.outcome_type, "info")

        getattr(st, outcome_color)(f"Outcome: {current_node.outcome}")
