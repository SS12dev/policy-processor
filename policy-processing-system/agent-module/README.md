# Policy Document Processor - Agent Module

**A2A-compliant stateless agent for policy document analysis and decision tree generation.**

---

## Overview

The agent module is a FastAPI-based A2A server that processes PDF policy documents using a LangGraph state machine. It extracts policies, generates decision trees, and validates structure—all while maintaining stateless operation through Redis.

### Key Features

- ✅ **A2A Protocol**: Google's Agent-to-Agent standard
- ✅ **Stateless Architecture**: Redis-based temporary storage
- ✅ **LangGraph Pipeline**: 8-node processing workflow
- ✅ **Multi-LLM Support**: OpenAI, Azure, LiteLLM Proxy
- ✅ **Horizontal Scaling**: Container-ready, concurrent requests
- ✅ **Real-time Streaming**: Progress updates via A2A events

---

## Quick Start

### 1. Installation

```bash
cd agent-module
pip install -r requirements.txt
```

### 2. Configuration

```bash
cp .env.example .env
nano .env  # Add your API keys
```

**Required Settings:**
```env
OPENAI_API_KEY=sk-...
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 3. Start Redis

```bash
# Docker
docker run -d -p 6379:6379 redis:latest

# Or local
redis-server
```

### 4. Run Agent Server

```bash
python -m a2a.server
```

Server starts on `http://localhost:8001`

---

## Directory Structure

```
agent-module/
├── a2a/                       # A2A Server Implementation
│   ├── __init__.py
│   ├── server.py              # FastAPI A2A application
│   ├── agent.py               # AgentExecutor with event streaming
│   └── redis_storage.py       # Redis state management
│
├── core/                      # LangGraph Processing Pipeline
│   ├── __init__.py
│   ├── langgraph_orchestrator.py  # Main orchestrator
│   ├── graph_nodes.py         # 8 processing nodes
│   ├── graph_state.py         # State definition
│   ├── policy_extractor.py    # Policy extraction (1526 lines)
│   ├── decision_tree_generator.py  # Tree generation (1192 lines)
│   ├── document_analyzer.py   # Document analysis
│   ├── document_verifier.py   # Verification stage
│   ├── pdf_processor.py       # PDF parsing
│   ├── chunking_strategy.py   # Smart chunking
│   ├── policy_aggregator.py   # Policy aggregation
│   ├── policy_refiner.py      # Refinement logic
│   ├── tree_validator.py      # Tree validation
│   ├── tree_generation_prompts.py  # LLM prompts
│   └── validator.py           # Overall validation
│
├── utils/                     # Utilities
│   ├── __init__.py
│   ├── llm.py                 # Async LLM client (multi-provider)
│   ├── logger.py              # Structured logging
│   └── redis_client.py        # Redis wrapper
│
├── models/                    # Data Models
│   ├── __init__.py
│   └── schemas.py             # Pydantic models
│
├── logs/                      # Runtime logs (created automatically)
├── metrics/                   # Metrics data (created automatically)
│
├── settings.py                # Pydantic settings
├── .env.example               # Configuration template
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Processing Pipeline

The LangGraph state machine executes 8 nodes:

```
1. PARSE_PDF
   ↓ Extract text, metadata, tables, images

2. ANALYZE_DOCUMENT
   ↓ Determine type, complexity, GPT-4 need

3. CHUNK_DOCUMENT
   ↓ Smart section-aware chunking

4. EXTRACT_POLICIES
   ↓ LLM-based policy extraction

5. GENERATE_TREES
   ↓ Decision tree creation

6. VALIDATE
   ↓ Structure & completeness checks
   ├─ RETRY (if validation fails)
   └─ Continue

7. VERIFY
   ↓ Document-level verification
   ├─ REFINE (if issues found)
   └─ COMPLETE

8. COMPLETE
   ↓ Aggregate results, return
```

---

## Configuration

### Environment Variables

All settings are configured via `.env` file or environment variables.

**Core Settings:**

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_HOST` | `0.0.0.0` | Server bind address |
| `SERVER_PORT` | `8001` | Server port |
| `LOG_LEVEL` | `INFO` | Logging level |

**Redis:**

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `localhost` | Redis server host |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_RESULT_TTL_HOURS` | `24` | Result storage TTL |

**LLM Provider:**

| Variable | Values | Description |
|----------|--------|-------------|
| `LLM_PROVIDER` | `openai`/`azure`/`proxy`/`auto` | LLM provider |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `OPENAI_MODEL_PRIMARY` | `gpt-4o-mini` | Primary model |
| `OPENAI_MODEL_SECONDARY` | `gpt-4o` | Secondary (GPT-4) model |

See [.env.example](.env.example) for all 50+ options.

---

## API Reference

### Agent Card

```bash
curl http://localhost:8001/.well-known/agent-card.json
```

### Process Document (A2A Protocol)

```python
import base64
from a2a.client import ClientFactory
from a2a import types

# Load PDF
with open("policy.pdf", "rb") as f:
    pdf_b64 = base64.b64encode(f.read()).decode()

# Create client
factory = ClientFactory()
client = factory.create(agent_url="http://localhost:8001")

# Send request
message = types.Message(
    message_id="msg_1",
    role=types.Role.user,
    parts=[types.Part(root=types.TextPart(text=json.dumps({
        "document_base64": pdf_b64,
        "use_gpt4": False,
        "confidence_threshold": 0.7
    })))]
)

# Stream events
async for event in client.stream(message):
    if isinstance(event, types.TaskStatusUpdateEvent):
        print(f"Status: {event.status.message.parts[0].root.text}")
    elif isinstance(event, types.TaskArtifactUpdateEvent):
        results = event.artifact.parts[0].root.data
        print(f"Policies: {results['policy_hierarchy']['total_policies']}")
```

### Input Schema

```json
{
  "document_base64": "string (base64)",
  "use_gpt4": "boolean (optional, default: false)",
  "enable_streaming": "boolean (optional, default: true)",
  "confidence_threshold": "number (optional, 0.0-1.0, default: 0.7)"
}
```

### Output Schema

```json
{
  "job_id": "string (UUID)",
  "status": "completed",
  "policy_hierarchy": {
    "total_policies": 15,
    "structure": [...]
  },
  "decision_trees": [...],
  "validation_result": {
    "is_valid": true,
    "issues": []
  },
  "processing_stats": {
    "processing_time_seconds": 45.2,
    "total_tokens": 12500
  }
}
```

---

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Adding Custom Nodes

1. **Define node function** in `core/graph_nodes.py`:

```python
def my_node(state: GraphState) -> GraphState:
    # Processing logic
    state.custom_field = process_data(state.document_text)
    return state
```

2. **Add to graph** in `core/langgraph_orchestrator.py`:

```python
graph.add_node("my_node", my_node)
graph.add_edge("previous_node", "my_node")
graph.add_edge("my_node", "next_node")
```

### Custom LLM Provider

The async LLM client supports multiple providers:

```python
from utils.llm import get_llm_client

# Get singleton client
client = get_llm_client()

# Direct async calls
result = await client.generate(
    prompt="Extract policies from...",
    use_gpt4=True
)

# Structured JSON output
data = await client.generate_structured(
    prompt="Extract as JSON...",
    use_gpt4=False
)

# LangChain integration
llm = client.get_langchain_model(use_gpt4=True)
```

---

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-m", "a2a.server"]
```

```bash
docker build -t policy-agent .
docker run -p 8001:8001 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e REDIS_HOST=redis \
  policy-agent
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: policy-agent
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: agent
        image: policy-agent:latest
        ports:
        - containerPort: 8001
        env:
        - name: REDIS_HOST
          value: "redis-service"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-creds
              key: api-key
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
```

---

## Monitoring

### Health Checks

```bash
# Readiness
curl http://localhost:8001/health

# Agent card
curl http://localhost:8001/.well-known/agent-card.json
```

### Metrics

Agent metrics are exported to `metrics/agent_metrics.json`:

```json
{
  "total_requests": 150,
  "successful_requests": 142,
  "failed_requests": 8,
  "avg_processing_time": 45.2,
  "active_jobs": 3
}
```

### Logs

Structured JSON logs in `logs/agent.log`:

```json
{
  "timestamp": "2025-12-12T10:30:00",
  "level": "INFO",
  "message": "Job completed",
  "job_id": "uuid",
  "processing_time": 45.2
}
```

---

## Troubleshooting

### Redis Connection Failed

```bash
# Check Redis is running
redis-cli ping  # Should return PONG

# Check connectivity
telnet localhost 6379
```

### LLM API Errors

```bash
# Verify API key
echo $OPENAI_API_KEY

# Test directly
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Processing Timeouts

Increase timeouts in `.env`:

```env
LLM_TIMEOUT=300
OPENAI_TIMEOUT=300
```

---

## License

MIT License - See LICENSE file

---

## Support

- **Issues**: https://github.com/your-repo/issues
- **Documentation**: https://docs.example.com
- **Email**: support@example.com
