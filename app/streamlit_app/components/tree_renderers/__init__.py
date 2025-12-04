"""
Tree renderers for decision tree and hierarchy visualization.

This package provides modular components for visualizing:
- Enhanced decision trees with hierarchical depth
- Policy hierarchies with parent-child relationships
- Color-coded outcomes and answer paths
"""

from .enhanced_renderer import (
    display_decision_tree_enhanced,
    render_tree_with_depth,
    render_node_recursive,
    render_question_tile,
    render_branch_with_outcome,
    render_outcome_tile,
    render_conditions_as_tiles
)

from .hierarchy_renderer import (
    display_policy_hierarchy_enhanced,
    render_policy_card,
    render_child_policy,
    render_grandchild_policy
)

__all__ = [
    # Enhanced decision tree renderer
    'display_decision_tree_enhanced',
    'render_tree_with_depth',
    'render_node_recursive',
    'render_question_tile',
    'render_branch_with_outcome',
    'render_outcome_tile',
    'render_conditions_as_tiles',
    
    # Hierarchy renderer
    'display_policy_hierarchy_enhanced',
    'render_policy_card',
    'render_child_policy',
    'render_grandchild_policy'
]
