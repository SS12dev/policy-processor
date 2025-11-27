"""
Enhanced prompts for decision tree generation with conditional routing.

These prompts guide the LLM to create decision trees with:
- Explicit routing logic
- Complete path coverage
- AND/OR conditional groups
- Proper outcome nodes
"""

DECISION_TREE_SYSTEM_PROMPT = """You are an expert at analyzing policy documents and creating comprehensive decision trees with conditional routing logic.

Your task is to convert policy requirements into a structured decision tree where:
1. Every question has EXPLICIT routing based on the answer
2. All possible answer paths lead to a clear outcome
3. Complex conditions use AND/OR logic groups
4. Each node has complete routing information

OUTPUT FORMAT:
You must return a valid JSON object with this exact structure:

{
  "root_node": {
    "node_id": "root",
    "node_type": "question",
    "question": {
      "question_id": "q1",
      "question_text": "Clear, specific question text",
      "question_type": "yes_no",
      "explanation": "Why this question determines eligibility",
      "source_references": [{
        "page_number": 1,
        "section": "Section title",
        "quoted_text": "Exact text from policy"
      }]
    },
    "children": {
      "yes": { /* next node object */ },
      "no": { /* next node object */ }
    },
    "confidence_score": 0.9
  }
}

NODE TYPES:
- "question": User-facing eligibility question
- "decision": System logic check (AND/OR conditions)
- "outcome": Terminal node (approved, denied, refer_to_manual, etc.)

QUESTION TYPES:
- "yes_no": Binary yes/no questions
- "multiple_choice": Select from options
- "numeric_range": Numeric value in range
- "date": Date-based questions

ROUTING RULES:
For YES/NO questions:
  "children": {
    "yes": { /* approval path or next question */ },
    "no": { /* denial path or alternative check */ }
  }

For MULTIPLE_CHOICE:
  "children": {
    "option_1_value": { /* path for option 1 */ },
    "option_2_value": { /* path for option 2 */ },
    "other": { /* fallback path */ }
  }

OUTCOME TYPES:
- "approved": Application meets all requirements
- "denied": Application fails requirements
- "refer_to_manual": Requires human review
- "pending_review": Needs additional documentation
- "requires_documentation": Missing required information

CRITICAL REQUIREMENTS:
1. EVERY question must have routing for ALL possible answers
2. EVERY path must eventually lead to an outcome node
3. Use decision nodes for complex AND/OR logic
4. Include source references with page numbers
5. Keep questions clear and unambiguous
6. Ensure logical flow matches policy intent

EXAMPLE:
{
  "root_node": {
    "node_id": "age_check",
    "node_type": "question",
    "question": {
      "question_id": "q1",
      "question_text": "Is the applicant 18 years or older?",
      "question_type": "yes_no",
      "explanation": "Policy requires applicants to be at least 18 years old",
      "source_references": [{
        "page_number": 1,
        "section": "Eligibility Criteria",
        "quoted_text": "Applicants must be at least 18 years of age"
      }]
    },
    "children": {
      "yes": {
        "node_id": "insurance_check",
        "node_type": "question",
        "question": {
          "question_id": "q2",
          "question_text": "Do you have existing health insurance?",
          "question_type": "yes_no",
          "explanation": "Policy covers both insured and uninsured individuals"
        },
        "children": {
          "yes": {
            "node_id": "outcome_approved",
            "node_type": "outcome",
            "outcome": "Approved - Meets age and insurance requirements",
            "outcome_type": "approved",
            "confidence_score": 0.95
          },
          "no": {
            "node_id": "outcome_approved_uninsured",
            "node_type": "outcome",
            "outcome": "Approved - Meets age requirement, eligible for uninsured coverage",
            "outcome_type": "approved",
            "confidence_score": 0.9
          }
        },
        "confidence_score": 0.92
      },
      "no": {
        "node_id": "outcome_denied_age",
        "node_type": "outcome",
        "outcome": "Denied - Must be at least 18 years old",
        "outcome_type": "denied",
        "source_references": [{
          "page_number": 1,
          "section": "Eligibility Criteria",
          "quoted_text": "Applicants must be at least 18 years of age"
        }],
        "confidence_score": 0.98
      }
    },
    "confidence_score": 0.95
  }
}

IMPORTANT: Ensure EVERY question node has routing for ALL possible answers. Never leave paths incomplete."""


AGGREGATOR_TREE_PROMPT_TEMPLATE = """Generate a ROUTING decision tree for this aggregator policy that determines which child policy applies.

Policy: {policy_title}
Description: {policy_description}

This is an AGGREGATOR policy with {num_children} child policies:
{child_list}

Create a decision tree that:
1. Asks questions to determine which child policy applies
2. Routes to the appropriate child policy based on answers
3. Handles cases where multiple policies might apply
4. Has clear routing logic for all scenarios

The tree should route to child policies using "router" type nodes:
{{
  "node_type": "router",
  "outcome": "Route to: {child_policy_title}",
  "outcome_type": "refer_to_manual",
  "child_policy_references": ["{child_policy_id}"]
}}

Source Document Context:
{context}

Generate a complete routing tree with:
- Questions that determine applicable child policy
- Clear routing for each answer path
- Fallback routing for edge cases
- Source references to policy text

Return only the JSON object, no additional text."""


LEAF_TREE_PROMPT_TEMPLATE = """Generate a comprehensive ELIGIBILITY decision tree for this policy.

Policy: {policy_title}
Description: {policy_description}
Level: {policy_level}
{parent_context}

Policy Conditions:
{conditions_text}

Create a decision tree that:
1. Asks questions to verify EACH condition
2. Routes based on answers to determine eligibility
3. Leads to clear outcomes (approved, denied, requires review)
4. Covers ALL policy requirements

Source Document Context:
{context}

REQUIREMENTS:
- Create one question per major condition
- Ensure all YES/NO questions have both yes and no paths
- Use decision nodes for complex AND/OR logic
- Every path must end in an outcome node
- Include exact quotes from policy document
- Assign appropriate outcome types

For AND conditions: All must be true for approval
For OR conditions: Any one being true allows approval

Generate a complete decision tree with proper routing.
Return only the JSON object, no additional text."""


def get_aggregator_prompt(
    policy_title: str,
    policy_description: str,
    child_policies: list,
    context: str
) -> str:
    """Generate prompt for aggregator tree creation."""
    child_list = "\n".join([
        f"- {getattr(child, 'title', 'Untitled')} (ID: {getattr(child, 'policy_id', 'unknown')})"
        for child in child_policies
    ])

    return AGGREGATOR_TREE_PROMPT_TEMPLATE.format(
        policy_title=policy_title,
        policy_description=policy_description,
        num_children=len(child_policies),
        child_list=child_list,
        context=context
    )


def get_leaf_prompt(
    policy_title: str,
    policy_description: str,
    policy_level: int,
    conditions: list,
    parent_context: str,
    context: str
) -> str:
    """Generate prompt for leaf tree creation."""
    conditions_text = "\n".join([
        f"{i+1}. {getattr(cond, 'description', str(cond))} ({getattr(cond, 'logic_type', 'AND')})"
        for i, cond in enumerate(conditions)
    ]) if conditions else "No explicit conditions listed"

    parent_info = f"Parent Policy: {parent_context}" if parent_context else "This is a root-level policy"

    return LEAF_TREE_PROMPT_TEMPLATE.format(
        policy_title=policy_title,
        policy_description=policy_description,
        policy_level=policy_level,
        parent_context=parent_info,
        conditions_text=conditions_text,
        context=context
    )


VALIDATION_PROMPT = """Review this decision tree for completeness and correctness:

Tree: {tree_json}

Check for:
1. Does every question have routing for ALL possible answers?
2. Does every path lead to an outcome node?
3. Are there any unreachable nodes?
4. Do AND/OR logic conditions make sense?
5. Are outcome types appropriate (approved/denied/review)?

If you find issues, suggest fixes in this format:
{{
  "is_valid": false,
  "issues": [
    {{
      "node_id": "q1",
      "issue": "Missing 'no' path",
      "suggestion": "Add routing for 'no' answer to appropriate outcome"
    }}
  ],
  "suggested_fixes": {{
    "q1": {{
      "add_child": {{
        "no": {{ "node_type": "outcome", "outcome_type": "denied" }}
      }}
    }}
  }}
}}

If valid, return:
{{
  "is_valid": true,
  "issues": [],
  "suggested_fixes": {{}}
}}"""
