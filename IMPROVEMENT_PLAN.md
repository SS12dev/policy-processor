# Policy Processor System Improvement Plan

## Overview
Comprehensive improvements to enhance decision tree generation, visualization, editing capabilities, and agent performance.

## Phase 1: Decision Tree Structure Enhancement

### Current State
- DecisionNode contains EligibilityQuestion objects
- Questions extracted into flat list for display
- Basic node types: question, outcome
- Limited conditional branching

### Required Improvements
1. **Conditional Branching Logic**
   - Each question node must route to different children based on answer
   - Support yes/no routing explicitly
   - Support multiple-choice routing with distinct paths
   - Support numeric range routing
   - Add routing_logic field to DecisionNode

2. **Answer-Based Navigation**
   - children dict: maps answer values to next node IDs
   - For yes/no: {"yes": node_id_1, "no": node_id_2}
   - For multiple choice: {option_value: node_id}
   - For numeric: {range_expression: node_id}

3. **Enhanced Node Types**
   - question: User-facing question with routing
   - decision: Logic/condition check (system-driven)
   - outcome: Terminal node (approved, denied, refer_to_manual)
   - router: Routes to child policy trees

### Schema Changes Needed
```python
class DecisionNode:
    node_id: str
    node_type: Literal["question", "decision", "outcome", "router"]
    question: Optional[EligibilityQuestion]
    decision_logic: Optional[str]  # For decision nodes
    outcome: Optional[str]
    outcome_type: Optional[str]
    children: Dict[str, str]  # answer_value -> child_node_id (currently full DecisionNode)
    routing_rules: Optional[Dict[str, Any]]  # Explicit routing configuration
    group_type: Optional[Literal["AND", "OR"]]  # For grouping related nodes
    parent_group: Optional[str]  # Group this node belongs to
```

## Phase 2: Visualization Improvements

### Current State
- Network graph using Plotly/NetworkX
- Shows nodes and edges generically
- Difficult to understand conditional flow

### Required: Tile-Based Vertical Tree View

#### Design Specifications
1. **Layout Structure**
   ```
   [Branch Dropdown]

   ┌─────────────────────────────────────┐
   │ ├─ Root Question                    │
   │ │                                   │
   │ ├── [YES] Next Question             │
   │ │   │                               │
   │ │   ├── [YES] Approved              │
   │ │   └── [NO] Check Condition X      │
   │ │                                   │
   │ └── [NO] Check Alternative Path     │
   │     │                               │
   │     └── [Condition Met] Refer       │
   └─────────────────────────────────────┘
   ```

2. **Visual Elements**
   - Vertical tiles with indentation for hierarchy
   - Left-side line connections showing flow
   - Different colors for node types:
     - Questions: Blue
     - Decisions: Yellow
     - Outcomes (Approved): Green
     - Outcomes (Denied): Red
     - Outcomes (Refer): Orange
   - Answer labels on connecting lines
   - AND/OR group indicators

3. **Interactive Features**
   - Collapsible/expandable branches
   - Click to highlight path
   - Hover for full details
   - Jump to node by ID

## Phase 3: Enhanced Editing Capabilities

### Current State
- Can edit questions and trees
- Basic add/remove operations
- Separate edit mode

### Required Improvements
1. **Node-Level Editing**
   - Add/remove/reorder nodes
   - Edit routing logic
   - Change node types
   - Set answer routing
   - Configure AND/OR groups

2. **Visual Tree Editor**
   - Drag-and-drop node reordering
   - Visual routing configuration
   - Inline editing in tree view
   - Path validation

3. **Bulk Operations**
   - Copy/paste branches
   - Duplicate trees
   - Merge similar questions
   - Export/import tree segments

## Phase 4: Agent Performance Optimization

### Current LangGraph Nodes
1. parse_pdf_node
2. analyze_document_node
3. chunk_document_node
4. extract_policy_node
5. generate_trees_node
6. validate_results_node
7. handle_retry_node
8. route_after_retry_node

### Optimization Opportunities

#### 1. parse_pdf_node
- **Current**: Basic PDF parsing
- **Improvement**: Add OCR fallback for scanned documents
- **Implementation**: Use pytesseract for image-based PDFs

#### 2. analyze_document_node
- **Current**: Rule-based document analysis
- **Improvement**: LLM-based document type detection
- **Benefit**: Better policy type recognition, more policy-agnostic
- **Implementation**: Add GPT-4o-mini call for quick classification

#### 3. chunk_document_node
- **Current**: Fixed-size chunks with overlap
- **Improvement**: Semantic chunking based on policy structure
- **Implementation**:
  - Use LLM to identify logical sections
  - Chunk by policy boundaries
  - Maintain full context for each policy section
  - Adaptive chunk sizing based on content complexity

#### 4. extract_policy_node
- **Current**: Single LLM call per chunk
- **Improvement**:
  - Parallel processing with batching
  - Schema-guided extraction
  - Progressive refinement
- **Implementation**:
  - Extract structure first (lightweight)
  - Then extract details (detailed)
  - Use structured outputs

#### 5. generate_trees_node
- **Current**: Generate trees independently
- **Improvement**:
  - Generate with explicit routing logic
  - Validate conditional paths
  - Ensure complete coverage
- **Implementation**:
  - Enhanced prompts for conditional logic
  - Path validation during generation
  - Retry incomplete branches

#### 6. validate_results_node
- **Current**: Basic validation checks
- **Improvement**:
  - Validate routing completeness
  - Check for unreachable nodes
  - Verify all outcomes are covered
  - Cross-reference with source document

#### 7. handle_retry_node
- **Current**: Retry low-confidence items
- **Improvement**:
  - Smarter routing based on error type
  - Selective retry (only problematic sections)
  - Progressive refinement vs full retry

#### 8. route_after_retry_node
- **Current**: Simple routing logic
- **Improvement**:
  - Route based on specific failure reasons
  - Skip successful components
  - Targeted re-extraction

### Chunking Strategy Improvements

#### Current Approach
- Fixed-size chunks (e.g., 2000 tokens)
- Fixed overlap (e.g., 200 tokens)
- No semantic awareness

#### Improved Approach
1. **Semantic Boundary Detection**
   ```python
   - Identify section headers
   - Detect policy boundaries
   - Find natural break points
   - Preserve context integrity
   ```

2. **Adaptive Sizing**
   ```python
   - Simple sections: Larger chunks
   - Complex sections: Smaller chunks
   - Tables/lists: Keep together
   - References: Include full context
   ```

3. **Hierarchical Chunking**
   ```python
   Level 1: Document sections
   Level 2: Policy groups
   Level 3: Individual policies
   Level 4: Conditions/criteria
   ```

4. **Context Preservation**
   ```python
   - Include parent policy context
   - Cross-reference related sections
   - Maintain definition awareness
   - Link to source pages
   ```

## Phase 5: Codebase Cleanup

### Files to Remove
- app/a2a/agent_old.py (unused, replaced by agent_refactored.py)
- Any temp/test files in root directory
- Unused migration scripts

### Files to Refactor
1. **app/a2a/agent_refactored.py** → **app/a2a/agent.py**
   - Rename to standard name
   - Update imports in server.py

2. **Consolidate schemas**
   - Review app/models/schemas.py
   - Remove unused model definitions
   - Add new routing-related models

### Code Standards
1. **Type Hints**: Ensure all functions have complete type hints
2. **Docstrings**: Google-style docstrings for all classes/functions
3. **Error Handling**: Consistent exception handling
4. **Logging**: Structured logging with consistent format
5. **Configuration**: Centralized settings management

## Implementation Priority

### Week 1: Core Infrastructure
- [ ] Update DecisionNode schema with routing
- [ ] Enhance decision tree generation prompts
- [ ] Implement path validation logic
- [ ] Add routing logic to tree generation

### Week 2: Visualization
- [ ] Design tile-based tree component
- [ ] Implement vertical tree layout
- [ ] Add collapsible branches
- [ ] Style node types differently
- [ ] Add AND/OR group indicators

### Week 3: Editing & Agent
- [ ] Build visual tree editor
- [ ] Add node routing configurator
- [ ] Optimize chunking strategy
- [ ] Improve document analyzer with LLM
- [ ] Enhance retry logic

### Week 4: Testing & Cleanup
- [ ] End-to-end testing
- [ ] Remove redundant files
- [ ] Code quality review
- [ ] Documentation update
- [ ] Performance benchmarking

## Success Metrics

### Decision Trees
- All questions have explicit routing
- No unreachable nodes
- 100% coverage of policy conditions
- Clear approval/denial paths

### Visualization
- Users can understand tree flow instantly
- Path from any question to outcome is clear
- AND/OR grouping is visually obvious

### Performance
- 30% faster processing
- 40% fewer retries needed
- More consistent chunk quality
- Better policy-agnostic handling

### Code Quality
- Zero redundant files
- 100% type hint coverage
- Comprehensive documentation
- All nodes logged properly
