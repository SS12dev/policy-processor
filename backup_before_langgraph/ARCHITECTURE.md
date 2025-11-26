# Policy Document Processor - Complete Architecture Guide

## Important Clarification

**This system is NOT using LangGraph.** It's a **sequential pipeline orchestrator** that processes documents through discrete stages. The confusion might arise from the name "orchestrator" and the use of LLMs, but this is a traditional ETL-style pipeline, not a LangGraph state machine.

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Processing Pipeline (The "Agent")](#processing-pipeline)
3. [A2A Protocol Wrapper](#a2a-protocol-wrapper)
4. [Data Flow](#data-flow)
5. [Component Details](#component-details)
6. [Streaming & Persistence](#streaming--persistence)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CLIENT LAYER                           │
│  (Streamlit UI, API Clients, other A2A-compliant agents)   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ JSON-RPC 2.0 over HTTP
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    A2A PROTOCOL LAYER                        │
│  ┌────────────────────────────────────────────────────┐    │
│  │  A2AStarletteApplication                           │    │
│  │  - Handles JSON-RPC requests                       │    │
│  │  - Routes to agent executor                        │    │
│  │  - Manages event queue for responses               │    │
│  └────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────┐    │
│  │  PolicyProcessorAgent (AgentExecutor)              │    │
│  │  - Parses A2A messages                             │    │
│  │  - Extracts parameters                             │    │
│  │  - Calls orchestrator                              │    │
│  │  - Sends responses via EventQueue                  │    │
│  └────────────────────────────────────────────────────┘    │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ Direct Python calls
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  ORCHESTRATOR LAYER                          │
│  ┌────────────────────────────────────────────────────┐    │
│  │  ProcessingOrchestrator                            │    │
│  │  - Sequential pipeline coordinator                 │    │
│  │  - 7-stage processing workflow                     │    │
│  │  - Status tracking & streaming                     │    │
│  │  - Result storage                                  │    │
│  └────────────────────────────────────────────────────┘    │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ Calls components in sequence
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 PROCESSING COMPONENTS                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ PDFProcessor │  │DocumentAnalyzer│ │ChunkingStrategy│   │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │PolicyExtractor│ │ TreeGenerator│  │  Validator   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ Stores/retrieves data
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    PERSISTENCE LAYER                         │
│  ┌────────────────┐         ┌────────────────────┐          │
│  │  Redis         │         │  SQLite/PostgreSQL │          │
│  │  - Status      │         │  - Jobs            │          │
│  │  - Results     │         │  - Documents       │          │
│  │  - Streaming   │         │  - Results         │          │
│  └────────────────┘         └────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

---

## Processing Pipeline (The "Agent")

### What We Mean by "Agent"

In this system, the term "agent" refers to the **entire processing pipeline wrapped in the A2A protocol**. It's not an agentic system with tool use and decision-making - it's a **deterministic sequential pipeline**.

### The 7 Processing Stages

The `ProcessingOrchestrator` coordinates a **sequential 7-stage pipeline**:

```python
Stage 1: PDF PARSING (0-10%)
    ↓
Stage 2: DOCUMENT ANALYSIS (10-20%)
    ↓
Stage 3: INTELLIGENT CHUNKING (20-30%)
    ↓
Stage 4: POLICY EXTRACTION (30-60%)
    ↓
Stage 5: DECISION TREE GENERATION (60-85%)
    ↓
Stage 6: VALIDATION (85-100%)
    ↓
Stage 7: COMPLETE (100%)
```

### Detailed Stage Breakdown

#### **Stage 1: PDF Parsing (0-10%)**

**Component**: `PDFProcessor`

**What it does**:
- Extracts text, tables, and images from PDF
- Parses document structure (headings, sections)
- Identifies numbered sections, bullet points
- Detects tables using pdfplumber

**Functions used**:
- `process_document(document_base64)` → Returns pages and metadata
- `extract_structure(pages)` → Returns structure dict

**Output**:
```python
pages: List[Page]  # Each page has text, tables, images
pdf_metadata: {
    "total_pages": int,
    "has_tables": bool,
    "has_images": bool
}
structure: {
    "has_numbered_sections": bool,
    "sections": List[Section],
    "tables": List[Table]
}
```

---

#### **Stage 2: Document Analysis (10-20%)**

**Component**: `DocumentAnalyzer`

**What it does**:
- Analyzes document complexity
- Determines document type (insurance, legal, corporate policy)
- Decides whether to use GPT-4 or GPT-4o-mini

**Functions used**:
- `analyze_document(pages, structure)` → Returns DocumentMetadata
- `should_use_gpt4(metadata, user_preference)` → Returns bool

**Output**:
```python
metadata: DocumentMetadata {
    document_type: Enum  # insurance_policy, legal_document, etc.
    complexity_score: float  # 0.0-1.0
    total_pages: int,
    has_images: bool,
    has_tables: bool,
    processing_time_seconds: float
}
```

**Decision Logic**:
- **Complexity > 0.7** OR **User requests GPT-4** → Use GPT-4 for extraction
- **Always use GPT-4** for decision tree generation (better quality)

---

#### **Stage 3: Intelligent Chunking (20-30%)**

**Component**: `ChunkingStrategy`

**What it does**:
- Splits document into semantic chunks
- Uses LLM to detect policy boundaries
- Respects natural breaks (sections, policies)
- Ensures chunks don't exceed token limits

**Functions used**:
- `chunk_document(pages, structure)` → Returns List[Chunk]
- `get_chunk_summary(chunks)` → Returns statistics

**Output**:
```python
chunks: List[Chunk] {
    chunk_id: str,
    text: str,
    page_range: (start, end),
    token_count: int,
    chunk_type: "policy" | "section" | "definition"
}
```

**Chunking Strategy**:
1. Identify policy boundaries using LLM
2. Group related content together
3. Split large policies if needed
4. Preserve context (include relevant surrounding text)

---

#### **Stage 4: Policy Extraction (30-60%)**

**Component**: `PolicyExtractor`

**What it does**:
- Extracts policies, sub-policies, and conditions from chunks
- Builds hierarchical policy structure
- Extracts definitions and references
- Uses structured output from LLM

**Functions used**:
- `extract_policies(chunks, pages)` → Returns PolicyHierarchy

**LLM Calls**:
- **Multiple parallel calls** (one per chunk)
- **Model**: GPT-4 or GPT-4o-mini (based on Stage 2 decision)
- **Structured output**: JSON schema with policies, conditions, definitions

**Output**:
```python
policy_hierarchy: PolicyHierarchy {
    root_policies: List[SubPolicy],
    definitions: Dict[str, str],
    total_policies: int,
    max_depth: int
}

SubPolicy {
    policy_id: str,
    title: str,
    description: str,
    level: int,  # Hierarchy level (0 = root)
    conditions: List[Condition],
    children: List[SubPolicy],  # Recursive hierarchy
    source_pages: List[int],
    source_text: str
}

Condition {
    condition_id: str,
    description: str,
    logic_type: "AND" | "OR" | "IF_THEN",
    required_fields: List[str],
    dependencies: List[str]
}
```

**How it builds hierarchy**:
1. Extract flat policies from each chunk
2. Analyze titles and levels to determine parent-child relationships
3. Build tree structure recursively
4. Resolve cross-references between policies

---

#### **Stage 5: Decision Tree Generation (60-85%)**

**Component**: `DecisionTreeGenerator`

**What it does**:
- Generates eligibility questions for each policy
- Creates decision trees with yes/no paths
- Ensures questions are clear and actionable
- Maintains context from parent policies

**Functions used**:
- `generate_hierarchical_trees(policy_hierarchy)` → Returns List[DecisionTree]
- `generate_tree_for_policy(policy)` → Returns DecisionTree

**LLM Calls**:
- **One call per policy** (parallelized with asyncio.gather)
- **Model**: Always GPT-4 (better quality)
- **Structured output**: Decision tree with questions and paths

**Output**:
```python
decision_trees: List[DecisionTree] {
    tree_id: str,
    policy_id: str,
    policy_title: str,
    root_node: TreeNode,
    total_nodes: int,
    total_paths: int,
    max_depth: int,
    confidence_score: float
}

TreeNode {
    node_id: str,
    node_type: "question" | "outcome" | "reference",
    question_text: str,  # For question nodes
    answer_type: "yes_no" | "multiple_choice",
    yes_branch: TreeNode,  # Recursive
    no_branch: TreeNode,   # Recursive
    outcome: str,  # For outcome nodes
    confidence: float
}
```

**How it works**:
1. For each policy in hierarchy:
   - Extract conditions and requirements
   - Generate eligibility questions
   - Build decision tree structure
   - Add yes/no branches
   - Add final outcomes (approved/denied/refer)
2. Maintain parent context for sub-policies
3. Ensure questions are non-overlapping and complete

---

#### **Stage 6: Validation (85-100%)**

**Component**: `Validator`

**What it does**:
- Validates completeness (all policies covered)
- Validates consistency (no contradictions)
- Validates traceability (questions map to source text)
- Checks confidence scores
- Identifies failed trees for retry

**Functions used**:
- `validate_all(policy_hierarchy, decision_trees, source_text)` → Returns ValidationResult

**LLM Calls**:
- **3 separate validation calls**: completeness, consistency, traceability
- **Model**: GPT-4
- **Structured output**: Issues, scores, recommendations

**Output**:
```python
validation_result: ValidationResult {
    is_valid: bool,
    overall_confidence: float,
    completeness_score: float,
    consistency_score: float,
    traceability_score: float,
    issues: List[Issue],
    sections_requiring_gpt4: List[str]  # Low-confidence trees to retry
}

Issue {
    issue_id: str,
    severity: "error" | "warning" | "info",
    category: "completeness" | "consistency" | "traceability",
    description: str,
    affected_sections: List[str],
    recommendation: str
}
```

**Retry Logic**:
- If validation fails with low confidence (<0.7):
  1. Identify low-confidence trees
  2. Retry those specific trees with GPT-4
  3. Re-validate with new trees
  4. Continue even if second validation fails

---

#### **Stage 7: Complete (100%)**

**What it does**:
- Packages all results into `ProcessingResponse`
- Stores results in Redis and SQLite
- Publishes completion event
- Returns final response

**Output**:
```python
response: ProcessingResponse {
    job_id: str,
    status: ProcessingStage.COMPLETED,
    metadata: DocumentMetadata,
    policy_hierarchy: PolicyHierarchy,
    decision_trees: List[DecisionTree],
    validation_result: ValidationResult,
    processing_stats: {
        "total_chunks": int,
        "total_tokens": int,
        "total_policies": int,
        "total_decision_trees": int,
        "used_gpt4_extraction": bool,
        "used_gpt4_trees": bool,
        "processing_time_seconds": float
    }
}
```

---

## A2A Protocol Wrapper

### What is A2A?

**A2A (Agent-to-Agent)** is a standardized communication protocol that allows different AI agents to discover and communicate with each other. Think of it like REST APIs, but specifically designed for AI agent interactions.

### Components of A2A Implementation

#### 1. **Agent Card** (Discovery Document)

Like an OpenAPI spec, the agent card describes the agent's capabilities:

```python
# Defined in: app/a2a/server.py

agent_card = {
    "name": "Policy Document Processor Agent",
    "version": "2.0.0",
    "description": "Process policy documents...",
    "url": "http://localhost:8001",

    "capabilities": {
        "streaming": True,      # Supports real-time updates
        "multiturn": False,     # Single-shot processing
        "push_notifications": False
    },

    "skills": [
        {
            "id": "process_policy",
            "name": "Process Policy Document",
            "description": "Upload and process...",

            # JSON Schema for inputs
            "input_schema": {
                "type": "object",
                "properties": {
                    "document_base64": {"type": "string"},
                    "use_gpt4": {"type": "boolean"},
                    "enable_streaming": {"type": "boolean"},
                    "confidence_threshold": {"type": "number"}
                }
            },

            # JSON Schema for outputs
            "output_schema": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "status": {"type": "string"},
                    "results": {"type": "object"}
                }
            }
        }
    ]
}
```

**Where it's served**: `http://localhost:8001/.well-known/agent-card`

---

#### 2. **Agent Executor** (Request Handler)

**Class**: `PolicyProcessorAgent` (extends `AgentExecutor`)

**File**: `app/a2a/agent.py`

**What it does**:
- Receives A2A requests via `execute()` method
- Parses JSON-RPC messages
- Extracts parameters from metadata
- Routes to appropriate handler (`_process_document` or `_get_results`)
- Calls the orchestrator
- Sends responses back through `EventQueue`

**Key Methods**:

```python
class PolicyProcessorAgent(AgentExecutor):

    async def execute(context: RequestContext, event_queue: EventQueue):
        # 1. Extract parameters from A2A message
        parameters = self._extract_parameters(context)

        # 2. Route request
        if "document_base64" in parameters:
            await self._process_document(context, event_queue, parameters)
        elif "job_id" in parameters:
            await self._get_results(context, event_queue, parameters["job_id"])

    async def _process_document(context, event_queue, parameters):
        # 1. Decode base64 PDF
        pdf_bytes = base64.b64decode(parameters["document_base64"])

        # 2. Create ProcessingRequest
        request = ProcessingRequest(
            document=parameters["document_base64"],
            processing_options={...}
        )

        # 3. Call orchestrator
        response = await self.orchestrator.process_document(request)

        # 4. Save to database
        self.db_ops.save_job(job_data)
        self.db_ops.save_document(document_data)
        self.db_ops.save_results(response.job_id, results_data)

        # 5. Send response via A2A event queue
        message = Message(
            role=Role.agent,
            parts=[TextPart(text="Processing complete!")],
            task_id=context.task_id
        )
        event_queue.enqueue_event(message)
```

---

#### 3. **Server Setup** (Starlette App)

**File**: `app/a2a/server.py`

**Components**:

```python
# 1. Create the agent executor
agent_executor = PolicyProcessorAgent(db_ops, orchestrator)

# 2. Create task store (tracks ongoing jobs)
task_store = InMemoryTaskStore()

# 3. Create request handler
request_handler = DefaultRequestHandler(
    agent_executor=agent_executor,
    task_store=task_store
)

# 4. Create agent card
agent_card = AgentCard.model_validate(create_agent_card())

# 5. Create A2A Starlette app
a2a_app = A2AStarletteApplication(
    agent_card=agent_card,
    http_handler=request_handler
)

# 6. Build ASGI app
app = a2a_app.build(
    agent_card_url="/.well-known/agent-card",
    rpc_url="/jsonrpc"
)

# 7. Run with Uvicorn
uvicorn.run(app, host="0.0.0.0", port=8001)
```

---

#### 4. **JSON-RPC Communication**

A2A uses **JSON-RPC 2.0** over HTTP:

**Request Format**:
```json
{
  "jsonrpc": "2.0",
  "method": "sendMessage",
  "params": {
    "message": {
      "messageId": "msg-123",
      "role": "user",
      "parts": [
        {
          "kind": "text",
          "text": "Process this document"
        }
      ],
      "taskId": "task-456",
      "contextId": "ctx-789"
    },
    "metadata": {
      "skill_id": "process_policy",
      "parameters": {
        "document_base64": "JVBERi0xLjQK...",
        "use_gpt4": false,
        "enable_streaming": true
      }
    }
  },
  "id": "req-001"
}
```

**Response Format**:
```json
{
  "jsonrpc": "2.0",
  "result": {
    "messages": [
      {
        "messageId": "msg-999",
        "role": "agent",
        "parts": [
          {
            "kind": "text",
            "text": "Processing complete!\n\nJob ID: abc-123\nStatus: completed"
          }
        ],
        "taskId": "task-456"
      }
    ]
  },
  "id": "req-001"
}
```

---

## Data Flow

### Complete Request Flow

```
1. Client sends PDF via Streamlit UI
   ↓
2. Streamlit app.py encodes PDF as base64
   ↓
3. A2AClientSync.process_document() creates JSON-RPC request
   ↓
4. HTTP POST to http://localhost:8001/jsonrpc
   ↓
5. A2AStarletteApplication receives request
   ↓
6. DefaultRequestHandler routes to PolicyProcessorAgent
   ↓
7. PolicyProcessorAgent.execute() called
   ↓
8. Extract parameters, decode base64
   ↓
9. Create ProcessingRequest object
   ↓
10. Call orchestrator.process_document()
    ↓
11. [7-STAGE PIPELINE RUNS]
    │
    ├─ Stage 1: PDFProcessor extracts text/tables
    ├─ Stage 2: DocumentAnalyzer analyzes type/complexity
    ├─ Stage 3: ChunkingStrategy splits into chunks
    ├─ Stage 4: PolicyExtractor extracts policies (LLM)
    ├─ Stage 5: TreeGenerator creates decision trees (LLM)
    ├─ Stage 6: Validator validates results (LLM)
    └─ Stage 7: Package and store results
    ↓
12. Return ProcessingResponse
    ↓
13. PolicyProcessorAgent saves to database
    ↓
14. Create A2A Message with results
    ↓
15. EventQueue.enqueue_event(message)
    ↓
16. JSON-RPC response sent to client
    ↓
17. A2AClientSync parses response
    ↓
18. Streamlit displays results
```

---

## Component Details

### This is NOT LangGraph

**What this system IS**:
- Sequential pipeline orchestrator
- Traditional ETL-style processing
- Deterministic flow (Stage 1 → Stage 2 → ... → Stage 7)
- Uses LLMs as tools within stages
- No state machine, no conditional routing

**What LangGraph would add** (if you wanted to convert):
- State management (`TypedDict` state)
- Conditional edges (dynamic routing)
- Cycles and feedback loops
- Human-in-the-loop approvals
- Multi-agent collaboration
- Persistent checkpoints

### Current "Nodes" (Processing Stages)

The 7 stages ARE analogous to "nodes" in a graph, but they're **hardcoded in sequence**, not defined in a graph structure:

```python
# Current implementation (Sequential)
async def process_document(request):
    # Stage 1 - hardcoded
    pages = pdf_processor.process_document(document)

    # Stage 2 - hardcoded
    metadata = await document_analyzer.analyze_document(pages)

    # Stage 3 - hardcoded
    chunks = await chunking_strategy.chunk_document(pages)

    # ... and so on
```

**If this were LangGraph**, it would look like:

```python
# Hypothetical LangGraph version
from langgraph.graph import StateGraph

# Define state
class ProcessingState(TypedDict):
    document: str
    pages: List[Page]
    chunks: List[Chunk]
    policies: PolicyHierarchy
    trees: List[DecisionTree]
    validation: ValidationResult

# Define nodes
async def parse_pdf_node(state: ProcessingState):
    pages, metadata = pdf_processor.process(state["document"])
    return {"pages": pages, "metadata": metadata}

async def analyze_node(state: ProcessingState):
    metadata = await analyzer.analyze(state["pages"])
    return {"metadata": metadata}

# ... more nodes

# Build graph
workflow = StateGraph(ProcessingState)
workflow.add_node("parse_pdf", parse_pdf_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("chunk", chunk_node)
# ...

# Add edges
workflow.add_edge("parse_pdf", "analyze")
workflow.add_edge("analyze", "chunk")
# ...

# Conditional routing (not in current system)
workflow.add_conditional_edges(
    "validate",
    should_retry,  # Function that decides
    {
        "retry": "generate_trees",  # Go back to stage 5
        "complete": "complete"       # Move forward
    }
)

# Compile
app = workflow.compile()
```

### Tools vs Custom Functions

**Current system**: No formal "tools" - just Python functions called directly

**What ARE "tools" in LLM context**:
- Functions the LLM can choose to call
- Defined with JSON schemas
- LLM decides when/how to use them
- Examples: web_search(), calculator(), database_query()

**In this system**:
- LLMs are used for **extraction only**
- No tool calling - LLMs just return structured JSON
- All functions are called explicitly by orchestrator

---

## Streaming & Persistence

### Streaming (Redis Pub/Sub)

**How it works**:

```python
# 1. Orchestrator publishes status updates
redis_client.publish(f"job:{job_id}:status", status_data)

# 2. Client subscribes to channel
pubsub = redis_client.pubsub()
pubsub.subscribe(f"job:{job_id}:status")

# 3. Client receives updates in real-time
while True:
    message = pubsub.get_message()
    if message:
        status = ProcessingStatus(**message["data"])
        print(f"Progress: {status.progress_percentage}%")
```

**What gets streamed**:
- Current stage (parsing, analyzing, extracting, etc.)
- Progress percentage (0-100%)
- Status message
- Errors (if any)

### Persistence

**Two storage systems**:

1. **Redis** (temporary, fast):
   - Job status (current progress)
   - Processing results (cached)
   - Streaming channels

2. **SQLite/PostgreSQL** (permanent):
   - Job metadata (`processing_jobs` table)
   - Original PDFs (`policy_documents` table)
   - Final results (`processing_results` table)
   - History log (`job_history` table)

**Data flow**:
```
Orchestrator
    ├─→ Redis (status updates during processing)
    └─→ SQLite (final results after completion)
         ↓
A2A Agent reads from SQLite
         ↓
Streamlit UI displays from SQLite
```

---

## Summary

### What This System IS:
✅ Sequential 7-stage document processing pipeline
✅ Wrapped in A2A protocol for standardized communication
✅ Uses LLMs for extraction, generation, and validation
✅ Streams progress via Redis Pub/Sub
✅ Stores results in SQLite/PostgreSQL
✅ Single unified endpoint via A2A

### What This System IS NOT:
❌ Not LangGraph (no state machine, no conditional routing)
❌ Not agentic (no tool calling, no autonomous decisions)
❌ Not multi-agent (single agent, single pipeline)
❌ Not using LangChain agents (just using LangChain's LLM wrappers)

### Key Takeaways:

1. **"Agent"** = The entire processing pipeline wrapped in A2A
2. **"Nodes"** = The 7 processing stages (hardcoded sequence)
3. **A2A** = Communication protocol wrapper (like REST API but for agents)
4. **Agent Card** = Declares skills and schemas (like OpenAPI spec)
5. **LLMs** = Used as tools within stages (not decision-makers)
6. **Streaming** = Redis Pub/Sub for real-time status updates
7. **Persistence** = SQLite for permanent storage, Redis for temporary cache

---

## Next Steps if You Want LangGraph

If you want to convert this to LangGraph:

1. Define `ProcessingState` TypedDict
2. Convert each stage to a node function
3. Add conditional edges for retry logic
4. Add human-in-the-loop approval points
5. Enable persistent checkpoints
6. Add multi-agent collaboration (e.g., separate extraction and validation agents)

But for your current use case, the sequential pipeline is simpler, faster, and more predictable!
