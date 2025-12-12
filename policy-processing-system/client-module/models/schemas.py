"""
Shared data models for client-side components.
Simplified version of agent models for UI rendering.
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum


class EligibilityQuestion(BaseModel):
    """An eligibility question in a decision tree."""
    question_id: str = Field(..., description="Unique question identifier")
    question_text: str = Field(..., description="The actual question to ask")
    question_type: str = Field(..., description="yes/no, multiple_choice, numeric, text")
    possible_answers: List[str] = Field(default_factory=list, description="Valid answers")
    explanation: str = Field(default="", description="Why this question is being asked")


class DecisionNode(BaseModel):
    """A node in the decision tree."""
    node_id: str = Field(..., description="Unique node identifier")
    node_type: str = Field(..., description="question, decision, outcome, router, group")

    # Node content based on type
    question: Optional[EligibilityQuestion] = Field(default=None, description="Question if this is a question node")
    decision_logic: Optional[str] = Field(default=None, description="Logic expression for decision nodes")
    outcome: Optional[str] = Field(default=None, description="Outcome if this is an outcome node")
    outcome_type: Optional[str] = Field(default=None, description="approved, denied, refer_to_manual, etc.")

    # Routing
    children: Dict[str, 'DecisionNode'] = Field(default_factory=dict, description="Child nodes keyed by answer")
    default_next_node_id: Optional[str] = Field(default=None, description="Default next node")

    # Metadata
    confidence_score: float = Field(default=0.0, description="Confidence in this node (0-1)")
    navigation_hint: Optional[str] = Field(default=None, description="Context hint for user")
    display_order: Optional[int] = Field(default=None, description="Order for display in UI")


class DecisionTree(BaseModel):
    """Complete decision tree for a policy."""
    tree_id: str = Field(..., description="Unique tree identifier")
    policy_id: str = Field(..., description="Associated policy ID")
    policy_title: str = Field(..., description="Policy title")
    root_node: DecisionNode = Field(..., description="Root node of the tree")
    questions: List[EligibilityQuestion] = Field(default_factory=list, description="All questions in the tree")

    # Tree statistics
    total_nodes: int = Field(default=0, description="Total number of nodes")
    total_paths: int = Field(default=0, description="Total number of possible paths")
    total_outcomes: int = Field(default=0, description="Number of outcome nodes")
    max_depth: int = Field(default=0, description="Maximum depth of the tree")
    confidence_score: float = Field(default=0.0, description="Overall confidence in tree (0-1)")

    # Validation
    has_complete_routing: bool = Field(default=False, description="Whether all paths lead to outcomes")
    unreachable_nodes: List[str] = Field(default_factory=list, description="Unreachable node IDs")


# Update forward references
DecisionNode.model_rebuild()
