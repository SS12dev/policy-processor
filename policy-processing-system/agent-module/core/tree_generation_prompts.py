"""
Prompts for decision tree generation with conditional routing.

These prompts guide the LLM to create decision trees with:
- Explicit routing logic
- Complete path coverage
- AND/OR conditional groups
- Proper outcome nodes
- **VALID JSON** (no trailing commas, proper formatting)
"""

DECISION_TREE_SYSTEM_PROMPT = """You are an expert at analyzing policy documents and creating comprehensive decision trees with conditional routing logic.

Your task is to convert policy requirements into a structured decision tree where:
1. Every question has EXPLICIT routing based on the answer
2. All possible answer paths lead to a clear outcome
3. Complex conditions use AND/OR logic groups
4. Each node has complete routing information

OUTPUT FORMAT:
You must return a **VALID JSON object** - NO trailing commas, NO comments, NO truncation.

CRITICAL JSON RULES:
- ❌ NO trailing commas: "section": "Title", ← WRONG
- ✅ Correct format: "section": "Title" ← RIGHT (no comma before })
- ❌ NO comments in JSON: // this is wrong
- ✅ Complete all brackets: every { must have matching }
- ❌ NO truncation: finish the entire JSON structure
- ✅ Validate JSON before returning

REQUIRED STRUCTURE:
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

QUESTION TYPES - Choose the MOST APPROPRIATE type for each requirement:

1. "yes_no" - Binary yes/no questions
   Use for: Simple binary decisions
   Example: "Is the patient 18 years or older?"
   Routing: {"yes": {...}, "no": {...}}

2. "numeric_range" - Numeric value questions with ranges
   Use for: Age, BMI, lab values, measurements, dosages
   Example: "What is the patient's BMI?"
   Routing: Based on value ranges {">=40": {...}, "35-39": {...}, "<35": {...}}

3. "multiple_choice" - Select one option from 3+ distinct choices
   Use for: Diagnosis types, procedure categories, treatment options
   Example: "What is the patient's diagnosis?"
   Routing: {"Type 1 Diabetes": {...}, "Type 2 Diabetes": {...}, "Gestational Diabetes": {...}}

4. "conditional" - ANY/ALL of the following requirements
   Use for: When policy says "any of the following" or "all of the following"
   Example: "Does the patient meet ANY of these criteria?"
   Routing: {"meets_criteria": {...}, "does_not_meet": {...}}

5. "date_based" - Time-sensitive or recency checks
   Use for: "within X months", "before/after date", waiting periods
   Example: "Has the patient had this procedure within the last 12 months?"
   Routing: {"within_timeframe": {...}, "outside_timeframe": {...}}

6. "categorical" - Classification or type selection
   Use for: Imaging types, surgical procedures, medication classes
   Example: "What type of imaging was performed?"
   Routing: {"CT Scan": {...}, "MRI": {...}, "Ultrasound": {...}, "X-Ray": {...}}

QUESTION TYPE SELECTION GUIDELINES:
- DEFAULT to yes_no ONLY for true binary decisions
- USE numeric_range for ANY numeric value (age, BMI, lab results, dosage)
- USE multiple_choice when policy lists 3+ specific options
- USE conditional when policy says "any of" or "all of the following"
- USE date_based for time/recency requirements
- USE categorical for procedure/diagnosis/treatment type selection

DO NOT use yes_no for everything - choose the type that best represents the policy logic!

ROUTING RULES:
For YES/NO:
  "children": {"yes": {...}, "no": {...}}

For NUMERIC_RANGE:
  "children": {">=40": {...}, "35-39": {...}, "<35": {...}}

For MULTIPLE_CHOICE:
  "children": {"option_1": {...}, "option_2": {...}, "option_3": {...}}

For CONDITIONAL:
  "children": {"meets_criteria": {...}, "does_not_meet": {...}}

For DATE_BASED:
  "children": {"within_timeframe": {...}, "outside_timeframe": {...}}

For CATEGORICAL:
  "children": {"category_1": {...}, "category_2": {...}, "other": {...}}

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
4. **Include source references with page numbers for ALL nodes** (questions AND outcomes)
5. Keep questions clear and unambiguous
6. Ensure logical flow matches policy intent
7. **RETURN VALID JSON** - check for trailing commas and complete all brackets

SOURCE REFERENCE REQUIREMENTS (CRITICAL FOR TRACEABILITY):
===========================================================
**QUESTION NODES** - MUST include source_references:
  - Reference the policy section that necessitates this question
  - Include page_number, section name, and quoted_text
  - Quote the exact policy language that requires this check

**OUTCOME NODES** - MUST include source_references:
  - Reference the policy section that defines this approval/denial/requirement
  - Include page_number, section name, and quoted_text
  - Quote the policy language that determines this outcome
  - THIS IS CRITICAL - outcome nodes without references reduce traceability score

Example QUESTION node with references:
{
  "node_type": "question",
  "question": {
    "question_text": "Is patient BMI >= 40?",
    "source_references": [{
      "page_number": 2,
      "section": "Eligibility Criteria",
      "quoted_text": "Bariatric surgery requires BMI >= 40 kg/m2"
    }]
  }
}

Example OUTCOME node with references (REQUIRED):
{
  "node_type": "outcome",
  "outcome": "APPROVED: Patient meets all bariatric surgery criteria",
  "outcome_type": "approved",
  "source_references": [{
    "page_number": 8,
    "section": "Coverage Determination",
    "quoted_text": "Coverage approved when patient meets BMI criteria, has documented weight loss history, and completes multidisciplinary regimen."
  }],
  "confidence_score": 0.95
}

DETAILED QUESTION TYPE EXAMPLES:
=================================
When analyzing a policy, choose the question type that BEST represents the requirement:

Example 1: Age requirement
Policy text: "Patient must be 65 years or older"
✓ GOOD: numeric_range - "What is the patient's age?" (precise, captures exact value)
✗ BAD:  yes_no - "Is patient 65 or older?" (loses precision)

Example 2: Multiple diagnoses
Policy text: "Covered diagnoses include: Type 1 DM, Type 2 DM, Gestational DM"
✓ GOOD: multiple_choice - "What is the diagnosis?" with options listed
✗ BAD:  yes_no for each (creates 3 questions instead of 1)

Example 3: ANY of the following
Policy text: "Patient must meet ANY of: BMI>=40, BMI>=35 with comorbidities, or failed other treatments"
✓ GOOD: conditional - "Does patient meet ANY of these criteria?" (list criteria)
✗ BAD:  yes_no for each (loses the ANY/OR logic)

Example 4: Time-based requirement
Policy text: "Must have had lab work within past 6 months"
✓ GOOD: date_based - "When was last lab work completed?"
✗ BAD:  yes_no - "Was lab work within 6 months?" (loses exact date)

Example 5: Procedure type
Policy text: "Covered imaging: CT, MRI, Ultrasound, or X-Ray"
✓ GOOD: categorical - "What type of imaging was performed?"
✗ BAD:  yes_no for each type (inefficient)

TREE DEPTH REQUIREMENTS (Target: 3-5 levels):
================================================
Break complex conditions into logical sequential steps for better granularity.

SHALLOW TREES (BAD - Depth 1-2):
✗ Single question → Outcome
✗ Misses intermediate decision steps
✗ Oversimplifies complex requirements

Example BAD tree:
Q1: "Does patient meet all criteria for bariatric surgery?"
  → YES: Approved
  → NO: Denied
Problem: "All criteria" should be broken into separate checks!

OPTIMAL TREES (GOOD - Depth 3-5):
✓ Break complex conditions into logical sequential steps
✓ Each level represents a distinct decision point
✓ Clear progression from broad → specific

Example GOOD tree:
Q1: "What is patient BMI?"
  → >=40: Q2: "Has patient completed required counseling?"
             → YES: Q3: "Are there any contraindications?"
                      → NO: Approved
                      → YES: Denied
             → NO: Requires Documentation
  → 35-39: Q2: "Does patient have documented comorbidities?"
              → YES: Q3: "Has patient completed counseling?"
                         (continue...)
              → NO: Denied
  → <35: Denied

DECOMPOSITION RULES:
1. AND Conditions → Sequential Questions (Deeper Tree)
   Policy: "Patient must have BMI >= 35 AND diabetes AND failed medication"
   Tree: Q1 (BMI) → Q2 (Diabetes) → Q3 (Medication failure)

2. OR Conditions → Branching Children (Wider Tree)
   Policy: "Approved if BMI >= 40 OR (BMI >= 35 AND comorbidities)"
   Tree: Q1 (BMI check) branches to different paths

3. Multi-Step Processes → Sequential Levels
   Policy: "Initial screening → Clinical assessment → Final approval"
   Tree: Q1 (Screening) → Q2 (Assessment) → Q3 (Approval criteria)

4. Compound Requirements → Break Into Atomic Steps
   Policy: "Coverage requires: (1) diagnosis, (2) failed treatment for 6+ months, (3) no contraindications"
   Tree: Q1 (Diagnosis) → Q2 (Treatment duration) → Q3 (Contraindications)

DEPTH TARGETS BY COMPLEXITY:
- Simple policy (1-2 requirements): 2-3 levels OK
- Standard policy (3-5 requirements): 3-4 levels recommended
- Complex policy (6+ requirements): 4-5 levels recommended
- Very complex multi-step approval: 5-6 levels acceptable

DO NOT CREATE:
✗ Single-level trees (question → outcome only)
✗ Compound questions that pack multiple checks into one
✗ Trees that skip intermediate decision logic

ALWAYS ASK YOURSELF:
✓ Can this question be broken into smaller steps?
✓ Are there intermediate checks between this and the outcome?
✓ Does the policy describe a multi-step process?

SIZE LIMITS:
- Keep trees FOCUSED and MANAGEABLE (max 8-10 questions)
- If policy is complex, prioritize the MOST IMPORTANT criteria
- Use outcome nodes with detailed explanations rather than endless questions
- NEVER truncate JSON - complete the entire structure properly
- Aim for 3-5 levels of depth for optimal granularity

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
          "explanation": "Policy covers both insured and uninsured individuals",
          "source_references": [{
            "page_number": 2,
            "section": "Insurance Requirements",
            "quoted_text": "Coverage is available for both insured and uninsured applicants"
          }]
        },
        "children": {
          "yes": {
            "node_id": "outcome_approved",
            "node_type": "outcome",
            "outcome": "Approved - Meets age and insurance requirements",
            "outcome_type": "approved",
            "source_references": [{
              "page_number": 3,
              "section": "Coverage Determination",
              "quoted_text": "Applicants who meet age and insurance requirements are approved for coverage"
            }],
            "confidence_score": 0.95
          },
          "no": {
            "node_id": "outcome_approved_uninsured",
            "node_type": "outcome",
            "outcome": "Approved - Meets age requirement, eligible for uninsured coverage",
            "outcome_type": "approved",
            "source_references": [{
              "page_number": 3,
              "section": "Coverage for Uninsured",
              "quoted_text": "Uninsured applicants meeting age requirements are eligible for coverage"
            }],
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


AGGREGATOR_TREE_PROMPT_TEMPLATE = """Generate a ROUTING decision tree for this aggregator policy.

=== AGGREGATOR POLICY ===
Policy Title: {policy_title}
Description: {policy_description}

This is an **AGGREGATOR** policy that routes to {num_children} child policies:
{child_list}

=== YOUR TASK ===
Create a simple ROUTING tree that:
1. **Asks 1-2 questions** to determine which child policy applies
2. **Routes to the appropriate child policy** based on the answer
3. **Handles edge cases** where no policy applies or multiple apply
4. **Uses clear, scenario-based questions** (not complex eligibility checks)

=== ROUTING STRATEGY ===
For aggregator policies, use **scenario-based routing**:
- Ask: "Which scenario best describes your situation?"
- Provide: Multiple choice with one option per child policy
- Route: Each option leads to a "refer_to_manual" outcome pointing to child policy

=== EXAMPLE STRUCTURE ===
{{
  "root_node": {{
    "node_id": "scenario_select",
    "node_type": "question",
    "question": {{
      "question_id": "q1",
      "question_text": "Which of the following scenarios best describes your situation?",
      "question_type": "multiple_choice",
      "explanation": "This policy covers multiple scenarios - please select the one that applies to you",
      "help_text": "Choose the option that most closely matches your specific situation",
      "options": [
        {{
          "option_id": "opt1",
          "label": "First Bariatric Surgery (No Prior Surgery)",
          "value": "first_surgery",
          "leads_to_node": "outcome_first_surgery"
        }},
        {{
          "option_id": "opt2",
          "label": "Revision of Prior Bariatric Surgery",
          "value": "revision_surgery",
          "leads_to_node": "outcome_revision"
        }},
        {{
          "option_id": "opt3",
          "label": "Adolescent Patient (Under 18)",
          "value": "adolescent",
          "leads_to_node": "outcome_adolescent"
        }}
      ]
    }},
    "children": {{
      "first_surgery": {{
        "node_id": "outcome_first_surgery",
        "node_type": "outcome",
        "outcome": "REFER TO: First Bariatric Surgery Policy - Review detailed eligibility criteria for initial surgery",
        "outcome_type": "refer_to_manual",
        "child_policy_references": ["bariatric_surgery_first_procedure"],
        "confidence_score": 0.90
      }},
      "revision_surgery": {{
        "node_id": "outcome_revision",
        "node_type": "outcome",
        "outcome": "REFER TO: Bariatric Surgery Revision Policy - Review criteria for revision procedures",
        "outcome_type": "refer_to_manual",
        "child_policy_references": ["bariatric_surgery_revision"],
        "confidence_score": 0.90
      }},
      "adolescent": {{
        "node_id": "outcome_adolescent",
        "node_type": "outcome",
        "outcome": "REFER TO: Adolescent Bariatric Surgery Policy - Special criteria apply for patients under 18",
        "outcome_type": "refer_to_manual",
        "child_policy_references": ["bariatric_surgery_adolescent"],
        "confidence_score": 0.90
      }}
    }},
    "confidence_score": 0.92
  }}
}}

=== CRITICAL GUIDELINES ===
1. **KEEP IT SIMPLE**: Routing trees should be 1-2 questions max
2. **SCENARIO-BASED**: Ask "which scenario" not detailed eligibility
3. **CLEAR OPTIONS**: Each child policy gets one clear option
4. **REFER OUTCOMES**: All outcomes should be "refer_to_manual" type
5. **CHILD REFERENCES**: Include "child_policy_references" array with child policy IDs
6. **NO TRAILING COMMAS**: Validate JSON structure

Source Document Context:
{context}

=== FINAL INSTRUCTIONS ===
1. Create a simple multiple-choice question with one option per child policy
2. Each option routes to an outcome that references the child policy
3. Add a fallback option for "None of the above" → manual review
4. Return ONLY the JSON object
5. **VALIDATE JSON** - no trailing commas!

Generate the routing tree now:"""


LEAF_TREE_PROMPT_TEMPLATE = """Generate a comprehensive ELIGIBILITY decision tree for this specific policy.

=== POLICY DETAILS ===
Policy Title: {policy_title}
Description: {policy_description}
Hierarchy Level: {policy_level}
{parent_context}

=== POLICY CONDITIONS ===
{conditions_text}

=== YOUR TASK ===
Create a complete decision tree that:
1. **Asks ONE eligibility question PER major condition**
2. **Routes based on YES/NO answers** to determine eligibility
3. **Leads to CLEAR OUTCOMES**: approved, denied, or requires_review
4. **Covers ALL policy requirements** from the conditions above
5. **Uses proper JSON structure** with NO trailing commas

=== IMPORTANT GUIDELINES ===
- START with the MOST IMPORTANT/FILTERING condition first (e.g., BMI threshold, age requirement)
- ASK SPECIFIC, MEASURABLE questions (not vague "do you meet criteria")
- For NUMERIC conditions: Use "numeric_range" question type with specific thresholds
- For DATE conditions: Use "date" question type
- For MULTIPLE criteria: Create separate questions for each
- For AND conditions: ALL must pass → ask sequentially, fail fast on first "no"
- For OR conditions: ANY can pass → ask each, route to approval on first "yes"

=== QUESTION QUALITY RULES ===
AVOID VAGUE QUESTIONS:
❌ BAD: "Do you meet the requirements?"
✅ GOOD: "Is your BMI 40 or greater, OR 35-39.9 with documented comorbidities?"

❌ BAD: "Have you tried losing weight?"
✅ GOOD: "Have you completed at least 6 months of medically supervised weight loss with less than 10% weight reduction?"

❌ BAD: "Does the prior surgery meet specific employment and coverage conditions?"
✅ GOOD: "Was the prior surgery performed while you were employed by your current employer AND covered under a benefit contract that counts toward the lifetime limit?"

❌ BAD: "Is medical management appropriate?"
✅ GOOD: "Have you attempted evidence-based medical management for at least 12 months, including diet modification, exercise program, and behavioral therapy, without achieving sustained weight loss?"

INCLUDE SPECIFICS:
- Timeframes: "6 months", "12 months", "within 30 days"
- Thresholds: "BMI ≥ 40", "Age 18-65", "10% weight reduction"
- Lists: "diabetes, hypertension, sleep apnea, or other comorbidities"
- Criteria: "medically supervised", "documented attempts", "FDA-approved"

ADD CONTEXT:
- explanation: Why this question matters for eligibility
- help_text: How to answer if unsure (e.g., "Ask your doctor", "Check your insurance card")
- validation_rules: For numeric/date questions, specify min/max/required

=== EXAMPLE STRUCTURE ===
For a policy with conditions: "BMI ≥40" AND "Age 18-65" AND "6 months supervised weight loss"

{{
  "root_node": {{
    "node_id": "bmi_check",
    "node_type": "question",
    "question": {{
      "question_id": "q1",
      "question_text": "What is your current Body Mass Index (BMI)?",
      "question_type": "numeric_range",
      "explanation": "Bariatric surgery requires a minimum BMI of 40 (or 35-39.9 with comorbidities)",
      "help_text": "BMI is calculated as weight (kg) / height (m)². Ask your doctor if unsure.",
      "source_references": [{{
        "page_number": 2,
        "section": "Medical Necessity Criteria",
        "quoted_text": "Body Mass Index of 40 or greater"
      }}],
      "validation_rules": {{
        "min": 0,
        "max": 100,
        "required": true
      }}
    }},
    "children": {{
      ">=40": {{
        "node_id": "age_check",
        "node_type": "question",
        "question": {{
          "question_id": "q2",
          "question_text": "What is your age?",
          "question_type": "numeric_range",
          "explanation": "Policy requires patients to be between 18 and 65 years old",
          "validation_rules": {{
            "min": 0,
            "max": 120,
            "required": true
          }}
        }},
        "children": {{
          "18-65": {{
            "node_id": "weight_loss_check",
            "node_type": "question",
            "question": {{
              "question_id": "q3",
              "question_text": "Have you completed at least 6 months of supervised weight loss attempts?",
              "question_type": "yes_no",
              "explanation": "Policy requires documented weight loss attempts before surgery"
            }},
            "children": {{
              "yes": {{
                "node_id": "outcome_approved",
                "node_type": "outcome",
                "outcome": "APPROVED: You meet all eligibility criteria for bariatric surgery",
                "outcome_type": "approved",
                "confidence_score": 0.95
              }},
              "no": {{
                "node_id": "outcome_denied_weight_loss",
                "node_type": "outcome",
                "outcome": "DENIED: Must complete 6 months of supervised weight loss before qualifying",
                "outcome_type": "denied",
                "confidence_score": 0.95
              }}
            }},
            "confidence_score": 0.92
          }},
          "<18": {{
            "node_id": "outcome_denied_too_young",
            "node_type": "outcome",
            "outcome": "DENIED: Must be at least 18 years old to qualify",
            "outcome_type": "denied",
            "confidence_score": 0.98
          }},
          ">65": {{
            "node_id": "outcome_review_age",
            "node_type": "outcome",
            "outcome": "REQUIRES REVIEW: Patients over 65 require additional medical assessment",
            "outcome_type": "requires_documentation",
            "confidence_score": 0.85
          }}
        }},
        "confidence_score": 0.93
      }},
      "<40": {{
        "node_id": "bmi_35_check",
        "node_type": "question",
        "question": {{
          "question_id": "q4",
          "question_text": "Is your BMI between 35 and 39.9?",
          "question_type": "yes_no",
          "explanation": "BMI 35-39.9 may qualify if you have comorbidities"
        }},
        "children": {{
          "yes": {{
            "node_id": "comorbidity_check",
            "node_type": "question",
            "question": {{
              "question_id": "q5",
              "question_text": "Do you have obesity-related comorbidities (diabetes, hypertension, sleep apnea)?",
              "question_type": "yes_no",
              "explanation": "BMI 35-39.9 requires comorbidities to qualify"
            }},
            "children": {{
              "yes": {{
                "node_id": "outcome_review_comorbid",
                "node_type": "outcome",
                "outcome": "REQUIRES REVIEW: BMI 35-39.9 with comorbidities - medical review needed",
                "outcome_type": "requires_documentation",
                "confidence_score": 0.80
              }},
              "no": {{
                "node_id": "outcome_denied_no_comorbid",
                "node_type": "outcome",
                "outcome": "DENIED: BMI 35-39.9 requires documented comorbidities",
                "outcome_type": "denied",
                "confidence_score": 0.92
              }}
            }},
            "confidence_score": 0.85
          }},
          "no": {{
            "node_id": "outcome_denied_bmi",
            "node_type": "outcome",
            "outcome": "DENIED: BMI must be at least 35 to qualify for bariatric surgery",
            "outcome_type": "denied",
            "confidence_score": 0.98
          }}
        }},
        "confidence_score": 0.90
      }}
    }},
    "confidence_score": 0.95
  }}
}}

=== CRITICAL REMINDERS ===
1. **NO TRAILING COMMAS**: Check every object before closing brace
2. **COMPLETE ALL PATHS**: Every YES/NO question needs BOTH yes AND no children
3. **END WITH OUTCOMES**: Every branch must eventually reach an outcome node
4. **CLEAR OUTCOME TYPES**:
   - "approved": Meets ALL requirements
   - "denied": Fails a requirement clearly stated in policy
   - "requires_documentation": Needs additional proof/documents
   - "pending_review": Edge case requiring human judgment
   - "refer_to_manual": Complex case not covered by tree
5. **INCLUDE SOURCE REFERENCES**: Add page_number, section, quoted_text where possible
6. **APPROPRIATE CONFIDENCE SCORES**:
   - 0.95-1.0: Clear yes/no criteria from policy
   - 0.85-0.94: Somewhat subjective or requires interpretation
   - 0.70-0.84: Ambiguous criteria or edge cases

Source Document Context:
{context}

=== FINAL INSTRUCTIONS ===
1. Read ALL policy conditions carefully
2. Create ONE question per major condition
3. Route logically based on answers
4. Ensure EVERY path has an outcome
5. Return ONLY the JSON object (no explanatory text before/after)
6. **VALIDATE your JSON** before returning - check for trailing commas!

Generate the complete decision tree now:"""


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

    # Combine system prompt with specific instructions
    specific_prompt = AGGREGATOR_TREE_PROMPT_TEMPLATE.format(
        policy_title=policy_title,
        policy_description=policy_description,
        num_children=len(child_policies),
        child_list=child_list,
        context=context
    )
    
    return f"{DECISION_TREE_SYSTEM_PROMPT}\n\n{specific_prompt}"


def get_leaf_prompt(
    policy_title: str,
    policy_description: str,
    policy_level: int,
    conditions: list,
    parent_context: str,
    context: str,
    source_references: list = None
) -> str:
    """
    Generate prompt for leaf tree creation.

    Args:
        policy_title: Title of the policy
        policy_description: Description of the policy
        policy_level: Hierarchy level (0=root, 1=child, etc.)
        conditions: List of PolicyCondition objects
        parent_context: Context about parent policy
        context: Additional context
        source_references: List of SourceReference objects from policy

    Returns:
        Complete prompt string
    """
    conditions_text = "\n".join([
        f"{i+1}. {getattr(cond, 'description', str(cond))} ({getattr(cond, 'logic_type', 'AND')})"
        for i, cond in enumerate(conditions)
    ]) if conditions else "No explicit conditions listed"

    parent_info = f"Parent Policy: {parent_context}" if parent_context else "This is a root-level policy"

    # Add source reference information to help LLM include page numbers
    source_info = ""
    if source_references and len(source_references) > 0:
        source_info = "\n\n=== SOURCE REFERENCES ===\n"
        source_info += "This policy is documented on the following pages:\n"
        for i, ref in enumerate(source_references[:5]):  # Limit to top 5 sources
            page = getattr(ref, 'page_number', 'N/A')
            section = getattr(ref, 'section_title', 'N/A')
            source_info += f"- Page {page}"
            if section and section != 'N/A':
                source_info += f", Section: {section}"
            source_info += "\n"
        source_info += "\n**IMPORTANT**: Include these page numbers in your source_references when generating questions.\n"
        source_info += "Example: \"source_references\": [{\"page_number\": " + str(getattr(source_references[0], 'page_number', 1)) + ", \"section\": \"" + getattr(source_references[0], 'section_title', 'Policy') + "\", \"quoted_text\": \"...\"}]\n"

    # Combine system prompt with specific instructions
    specific_prompt = LEAF_TREE_PROMPT_TEMPLATE.format(
        policy_title=policy_title,
        policy_description=policy_description,
        policy_level=policy_level,
        parent_context=parent_info,
        conditions_text=conditions_text,
        context=context + source_info
    )

    return f"{DECISION_TREE_SYSTEM_PROMPT}\n\n{specific_prompt}"


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
