"""
Tree renderers for decision tree and hierarchy visualization.

This package provides modular components for visualizing:
- Decision trees with hierarchical depth
- Policy hierarchies with parent-child relationships
- Color-coded outcomes and answer paths
"""

from .decision_tree_renderer import (
    display_decision_tree,
    render_tree_with_depth,
    render_node_recursive,
    render_question_tile,
    render_branch_with_outcome,
    render_outcome_tile,
    render_conditions_as_tiles
)

from .hierarchy_renderer import (
    display_policy_hierarchy,
    render_policy_card,
    render_child_policy,
    render_grandchild_policy
)

__all__ = [
    # Decision tree renderer
    'display_decision_tree',
    'render_tree_with_depth',
    'render_node_recursive',
    'render_question_tile',
    'render_branch_with_outcome',
    'render_outcome_tile',
    'render_conditions_as_tiles',
    
    # Hierarchy renderer
    'display_policy_hierarchy',
    'render_policy_card',
    'render_child_policy',
    'render_grandchild_policy'
]
