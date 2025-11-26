# Visual Flow Diagrams

## 1. Complete System Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                          USER INTERACTION                             │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                    User uploads PDF in Streamlit
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI (Port 8501)                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  app/streamlit_app/app.py                                    │  │
│  │  - Tab 1: Upload & Process                                   │  │
│  │  - Tab 2: Review Decision Trees                              │  │
│  │  - Encodes PDF to base64                                     │  │
│  │  - Calls A2A client                                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                    a2a_client.process_document()
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    A2A CLIENT (Sync Wrapper)                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  app/streamlit_app/a2a_client.py                             │  │
│  │  - A2AClientSync wraps async A2AClient                       │  │
│  │  - Creates JSON-RPC 2.0 request                              │  │
│  │  - Sends POST to http://localhost:8001/jsonrpc               │  │
│  └──────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                        HTTP POST (JSON-RPC)
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    A2A SERVER (Port 8001)                            │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  app/a2a/server.py                                           │  │
│  │  - A2AStarletteApplication (ASGI)                            │  │
│  │  - Serves agent card at /.well-known/agent-card              │  │
│  │  - Routes /jsonrpc to DefaultRequestHandler                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                    DefaultRequestHandler routes request
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    A2A AGENT EXECUTOR                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  app/a2a/agent.py - PolicyProcessorAgent                     │  │
│  │  ┌──────────────────────────────────────────────────────┐   │  │
│  │  │  execute(context, event_queue)                       │   │  │
│  │  │  1. Extract parameters from context                  │   │  │
│  │  │  2. Decode base64 PDF                                │   │  │
│  │  │  3. Create ProcessingRequest                         │   │  │
│  │  │  4. Call orchestrator                                │   │  │
│  │  │  5. Save results to database                         │   │  │
│  │  │  6. Send response via EventQueue                     │   │  │
│  │  └──────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
            orchestrator.process_document(request)
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PROCESSING ORCHESTRATOR                           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  app/core/orchestrator.py - ProcessingOrchestrator          │  │
│  │  ┌──────────────────────────────────────────────────────┐   │  │
│  │  │  process_document(request) - Sequential Pipeline    │   │  │
│  │  │                                                      │   │  │
│  │  │  Stage 1: PDF Parsing           [===   ] 10%        │   │  │
│  │  │  Stage 2: Document Analysis     [=====  ] 20%       │   │  │
│  │  │  Stage 3: Chunking              [======= ] 30%      │   │  │
│  │  │  Stage 4: Policy Extraction     [========] 60%      │   │  │
│  │  │  Stage 5: Tree Generation       [=========] 85%     │   │  │
│  │  │  Stage 6: Validation            [==========] 100%   │   │  │
│  │  │  Stage 7: Complete              [==========] DONE   │   │  │
│  │  └──────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└──┬─────┬─────┬─────┬─────┬─────┬─────┬─────────────────────────────┘
   │     │     │     │     │     │     │
   │     │     │     │     │     │     └─→ Publishes to Redis
   │     │     │     │     │     │         redis.publish("job:123:status", ...)
   │     │     │     │     │     │
   │     │     │     │     │     └───────→ Validator
   │     │     │     │     │               (LLM validation)
   │     │     │     │     │
   │     │     │     │     └─────────────→ DecisionTreeGenerator
   │     │     │     │                     (GPT-4 tree generation)
   │     │     │     │
   │     │     │     └───────────────────→ PolicyExtractor
   │     │     │                           (GPT-4/mini extraction)
   │     │     │
   │     │     └─────────────────────────→ ChunkingStrategy
   │     │                                 (LLM-assisted chunking)
   │     │
   │     └───────────────────────────────→ DocumentAnalyzer
   │                                       (Type/complexity analysis)
   │
   └─────────────────────────────────────→ PDFProcessor
                                           (PDF parsing)
```

---

## 2. A2A Protocol Flow (JSON-RPC)

```
CLIENT                           SERVER
  │                                │
  │  1. Encode PDF to base64       │
  │     "JVBERi0xLjQK..."          │
  │                                │
  │  2. Create JSON-RPC request    │
  │     POST /jsonrpc              │
  ├────────────────────────────────>
  │     {                          │
  │       "jsonrpc": "2.0",        │
  │       "method": "sendMessage", │
  │       "params": {              │
  │         "message": {...},      │
  │         "metadata": {          │
  │           "skill_id": "process_policy",
  │           "parameters": {      │
  │             "document_base64": "...",
  │             "use_gpt4": false, │
  │             ...                │
  │           }                    │
  │         }                      │
  │       },                       │
  │       "id": "req-001"          │
  │     }                          │
  │                                │
  │  3. Server routes to agent     │
  │                                │───> PolicyProcessorAgent.execute()
  │                                │         │
  │                                │         ▼
  │                                │     Extract parameters
  │                                │         │
  │                                │         ▼
  │                                │     Decode PDF
  │                                │         │
  │                                │         ▼
  │                                │     Call orchestrator.process_document()
  │                                │         │
  │                                │         ▼
  │                                │     [7-STAGE PIPELINE RUNS]
  │                                │         │
  │  4. Receive status updates     │         ▼
  │     (via Redis Pub/Sub)        │     Save to database
  │<════════════════════════════════         │
  │     Progress: 10%              │         ▼
  │<════════════════════════════════     Create A2A Message
  │     Progress: 30%              │         │
  │<════════════════════════════════         ▼
  │     Progress: 60%              │     EventQueue.enqueue_event()
  │<════════════════════════════════
  │     Progress: 100%             │
  │                                │
  │  5. Receive final response     │
  │     200 OK                     │
  │<────────────────────────────────
  │     {                          │
  │       "jsonrpc": "2.0",        │
  │       "result": {              │
  │         "messages": [          │
  │           {                    │
  │             "role": "agent",   │
  │             "parts": [         │
  │               {                │
  │                 "kind": "text",│
  │                 "text": "Processing complete!\n\nJob ID: abc-123..."
  │               }                │
  │             ]                  │
  │           }                    │
  │         ]                      │
  │       },                       │
  │       "id": "req-001"          │
  │     }                          │
  │                                │
  │  6. Parse response             │
  │     Extract job_id             │
  │                                │
```

---

## 3. Processing Pipeline (7 Stages)

```
INPUT: PDF (base64)
   │
   ▼
┌─────────────────────────────────────────┐
│ STAGE 1: PDF PARSING (0-10%)           │
│ Component: PDFProcessor                 │
│ ┌─────────────────────────────────────┐ │
│ │ - Extract text from each page       │ │
│ │ - Parse tables with pdfplumber      │ │
│ │ - Extract images                    │ │
│ │ - Identify document structure       │ │
│ │   (headings, sections, bullets)     │ │
│ └─────────────────────────────────────┘ │
│ Output: List[Page], Structure, Metadata │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ STAGE 2: DOCUMENT ANALYSIS (10-20%)    │
│ Component: DocumentAnalyzer             │
│ ┌─────────────────────────────────────┐ │
│ │ - Analyze document type             │ │
│ │ - Calculate complexity score        │ │
│ │ - Determine LLM model to use        │ │
│ │   (GPT-4 vs GPT-4o-mini)            │ │
│ └─────────────────────────────────────┘ │
│ Output: DocumentMetadata, Model Choice  │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ STAGE 3: CHUNKING (20-30%)             │
│ Component: ChunkingStrategy             │
│ ┌─────────────────────────────────────┐ │
│ │ - LLM detects policy boundaries     │ │
│ │ - Split into semantic chunks        │ │
│ │ - Respect token limits              │ │
│ │ - Preserve context                  │ │
│ └─────────────────────────────────────┘ │
│ Output: List[Chunk] (~10-20 chunks)     │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ STAGE 4: POLICY EXTRACTION (30-60%)    │
│ Component: PolicyExtractor              │
│ LLM: GPT-4 or GPT-4o-mini               │
│ ┌─────────────────────────────────────┐ │
│ │ FOR EACH CHUNK (parallel):          │ │
│ │   - Extract policies                │ │
│ │   - Extract conditions              │ │
│ │   - Extract definitions             │ │
│ │   - Identify hierarchy              │ │
│ │                                     │ │
│ │ THEN:                               │ │
│ │   - Build hierarchical structure    │ │
│ │   - Resolve cross-references        │ │
│ └─────────────────────────────────────┘ │
│ Output: PolicyHierarchy                 │
│   - ~5-20 root policies                 │
│   - ~10-50 total policies               │
│   - ~50-200 conditions                  │
│   - ~10-30 definitions                  │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ STAGE 5: TREE GENERATION (60-85%)      │
│ Component: DecisionTreeGenerator        │
│ LLM: Always GPT-4                       │
│ ┌─────────────────────────────────────┐ │
│ │ FOR EACH POLICY (parallel):         │ │
│ │   - Generate eligibility questions  │ │
│ │   - Create yes/no decision paths    │ │
│ │   - Add final outcomes              │ │
│ │   - Calculate confidence scores     │ │
│ └─────────────────────────────────────┘ │
│ Output: List[DecisionTree]              │
│   - One tree per policy                 │
│   - ~3-10 questions per tree            │
│   - ~2-5 levels deep                    │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ STAGE 6: VALIDATION (85-100%)          │
│ Component: Validator                    │
│ LLM: GPT-4                              │
│ ┌─────────────────────────────────────┐ │
│ │ 1. Completeness check               │ │
│ │    - All policies covered?          │ │
│ │    - All conditions in trees?       │ │
│ │                                     │ │
│ │ 2. Consistency check                │ │
│ │    - No contradictions?             │ │
│ │    - Logic flows correctly?         │ │
│ │                                     │ │
│ │ 3. Traceability check               │ │
│ │    - Questions match source text?   │ │
│ │    - Accurate references?           │ │
│ │                                     │ │
│ │ 4. If confidence < 0.7:             │ │
│ │    - Identify failed trees          │ │
│ │    - Retry with GPT-4               │ │
│ │    - Re-validate                    │ │
│ └─────────────────────────────────────┘ │
│ Output: ValidationResult                │
│   - Overall confidence score            │
│   - List of issues                      │
│   - Pass/fail status                    │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ STAGE 7: COMPLETE (100%)                │
│ ┌─────────────────────────────────────┐ │
│ │ - Package all results               │ │
│ │ - Store in Redis (cache)            │ │
│ │ - Store in SQLite (permanent)       │ │
│ │ - Publish completion event          │ │
│ │ - Return ProcessingResponse         │ │
│ └─────────────────────────────────────┘ │
│ Output: ProcessingResponse              │
└───────────────┬─────────────────────────┘
                │
                ▼
OUTPUT: Complete results with policy hierarchy and decision trees
```

---

## 4. Data Storage Flow

```
┌────────────────────────────────────────────────┐
│           DURING PROCESSING                     │
│                                                 │
│  Orchestrator publishes to Redis:               │
│  ┌───────────────────────────────────────┐     │
│  │ Channel: job:abc-123:status           │     │
│  │                                       │     │
│  │ {                                     │     │
│  │   "job_id": "abc-123",                │     │
│  │   "stage": "EXTRACTING_POLICIES",     │     │
│  │   "progress_percentage": 45,          │     │
│  │   "message": "Extracting policies..." │     │
│  │ }                                     │     │
│  └───────────────────────────────────────┘     │
│           │                                     │
│           ▼                                     │
│  Client subscribes and receives updates        │
└────────────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────┐
│           AFTER COMPLETION                      │
│                                                 │
│  1. Redis (Temporary Cache):                   │
│     ┌─────────────────────────────────┐        │
│     │ Key: job:abc-123:result         │        │
│     │ Value: {                        │        │
│     │   "job_id": "abc-123",          │        │
│     │   "status": "completed",        │        │
│     │   "policy_hierarchy": {...},    │        │
│     │   "decision_trees": [...]       │        │
│     │ }                               │        │
│     │ TTL: 24 hours                   │        │
│     └─────────────────────────────────┘        │
│                                                 │
│  2. SQLite (Permanent Storage):                │
│     ┌─────────────────────────────────┐        │
│     │ TABLE: processing_jobs          │        │
│     │ ├─ job_id: "abc-123"            │        │
│     │ ├─ status: "completed"          │        │
│     │ ├─ created_at: "2025-11-25..."  │        │
│     │ └─ ...                          │        │
│     ├─────────────────────────────────┤        │
│     │ TABLE: policy_documents         │        │
│     │ ├─ job_id: "abc-123"            │        │
│     │ ├─ content_base64: "JVBERi..."  │        │
│     │ ├─ filename: "policy.pdf"       │        │
│     │ └─ ...                          │        │
│     ├─────────────────────────────────┤        │
│     │ TABLE: processing_results       │        │
│     │ ├─ job_id: "abc-123"            │        │
│     │ ├─ policy_hierarchy_json: {...} │        │
│     │ ├─ decision_trees_json: [...]   │        │
│     │ └─ validation_result_json: {...}│        │
│     └─────────────────────────────────┘        │
└────────────────────────────────────────────────┘
```

---

## 5. LLM Usage Pattern

```
┌──────────────────────────────────────────────────────┐
│              LLM CALLS IN PIPELINE                   │
└──────────────────────────────────────────────────────┘

STAGE 3: Chunking
   │
   ├─> LLM Call #1: Detect policy boundaries
   │   Model: GPT-4o-mini
   │   Input: Full document text
   │   Output: List of policy start/end positions
   │
   ▼

STAGE 4: Policy Extraction
   │
   ├─> LLM Call #2-11: Extract policies (10 chunks)
   │   Model: GPT-4 or GPT-4o-mini (based on complexity)
   │   Input: Chunk text (each ~2000 tokens)
   │   Output: Policies, conditions, definitions (JSON)
   │   Parallel: Yes (asyncio.gather)
   │
   ▼

STAGE 5: Tree Generation
   │
   ├─> LLM Call #12-31: Generate trees (20 policies)
   │   Model: Always GPT-4
   │   Input: Policy details + hierarchy context
   │   Output: Decision tree with questions (JSON)
   │   Parallel: Yes (asyncio.gather)
   │
   ▼

STAGE 6: Validation
   │
   ├─> LLM Call #32: Completeness validation
   │   Model: GPT-4
   │   Input: Policies + trees + source text
   │   Output: Completeness issues (JSON)
   │
   ├─> LLM Call #33: Consistency validation
   │   Model: GPT-4
   │   Input: All policies and trees
   │   Output: Consistency issues (JSON)
   │
   ├─> LLM Call #34: Traceability validation
   │   Model: GPT-4
   │   Input: Trees + source text
   │   Output: Traceability issues (JSON)
   │
   └─> IF low confidence detected:
       │
       ├─> LLM Call #35-39: Retry failed trees (5 retries)
       │   Model: GPT-4
       │   Input: Failed policy details
       │   Output: Improved decision tree (JSON)
       │
       └─> LLM Call #40-42: Re-validate
           Model: GPT-4
           Input: Updated trees + source
           Output: New validation results (JSON)

TOTAL LLM CALLS: ~30-45 per document
TOTAL COST: ~$0.50-$2.00 per document (depending on length and complexity)
TOTAL TIME: ~2-10 minutes
```

---

## Key Insights

### Why NOT LangGraph?

1. **Linear workflow**: No branching logic needed
2. **Predictable**: Always follows same 7 stages
3. **Simpler**: Easier to debug and maintain
4. **Faster**: No state management overhead

### When TO use LangGraph?

- **Conditional routing**: Different paths based on document type
- **Human-in-the-loop**: Need approval before proceeding
- **Multi-agent**: Multiple specialized agents collaborating
- **Iterative refinement**: Cycle back to improve results
- **Complex decision trees**: Multiple possible workflows

### Current System Strengths

✅ Simple, predictable flow
✅ Easy to debug (linear execution)
✅ Fast (no state management overhead)
✅ Reliable (deterministic)
✅ Production-ready

### Potential Improvements with LangGraph

- Add conditional retry logic (smart decisions on when to retry)
- Human approval before expensive GPT-4 calls
- Multiple extraction strategies based on document type
- Iterative refinement loops
- Persistent checkpoints for long-running jobs
