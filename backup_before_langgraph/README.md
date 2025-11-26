# Policy Document Processor

AI-powered policy document processor that extracts policies, generates decision trees, and validates structure using A2A protocol.

## Features

- **Single A2A Endpoint** - Unified policy processing via A2A protocol
- **PDF Storage** - Original documents stored for review
- **Decision Trees** - Automatic generation with eligibility questions
- **Streaming Support** - Backend-controlled real-time progress updates
- **Side-by-Side Review** - View PDFs alongside generated decision trees
- **2-Tab Interface** - Clean Streamlit UI for upload and review

## Quick Start

### Prerequisites

1. Python 3.11+
2. OpenAI API key
3. Redis server (optional, for caching)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="sk-..."
```

### Running the Application

**Step 1: Start the A2A Server**

```bash
python main_a2a_simplified.py
```

Server will be available at:
- Agent endpoint: `http://localhost:8001`
- Agent card: `http://localhost:8001/.well-known/agent-card`
- Health check: `http://localhost:8001/health`

**Step 2: Start the Streamlit UI** (in a new terminal)

```bash
python main_streamlit_simplified.py
```

UI will open at: `http://localhost:8501`

### Usage

1. **Upload & Process Tab**
   - Upload PDF policy document
   - Configure processing options:
     - Use GPT-4 (higher accuracy, slower)
     - Enable streaming (real-time updates)
     - Set confidence threshold (0.0-1.0)
   - Click "Process Document"
   - Wait for completion (2-10 minutes)

2. **Review Decision Trees Tab**
   - Enter Job ID (auto-populated from processing)
   - Click "Load Results"
   - Choose view mode:
     - Decision Trees Only
     - Side-by-Side (PDF + Trees)
   - Explore results and export as JSON

## Architecture

```
┌─────────────────────────┐
│   Streamlit Frontend    │ Port 8501
│  - Upload & Process     │
│  - Review Trees         │
└───────────┬─────────────┘
            │ JSON-RPC 2.0
            ▼
┌─────────────────────────┐
│    A2A Agent Server     │ Port 8001
│  - Single endpoint      │
│  - Streaming enabled    │
│  - Direct orchestrator  │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│    SQLite Database      │
│  - Jobs                 │
│  - Documents (PDFs)     │
│  - Results              │
│  - History              │
└─────────────────────────┘
```

## Project Structure

```
policy processor/
├── main_a2a_simplified.py          # A2A server entry point
├── main_streamlit_simplified.py    # Streamlit UI entry point
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── app/
│   ├── a2a/
│   │   ├── simplified_agent.py     # A2A agent executor
│   │   └── simplified_server.py    # A2A server setup
│   │
│   ├── core/
│   │   ├── orchestrator.py         # Main processing orchestrator
│   │   ├── pdf_processor.py        # PDF extraction
│   │   ├── policy_extractor.py     # Policy extraction with LLM
│   │   ├── policy_aggregator.py    # Policy hierarchy builder
│   │   ├── decision_tree_generator.py # Decision tree generation
│   │   ├── validator.py            # Result validation
│   │   ├── document_analyzer.py    # Document analysis
│   │   └── chunking_strategy.py    # Document chunking
│   │
│   ├── database/
│   │   ├── models.py               # SQLAlchemy models
│   │   └── operations.py           # Database CRUD operations
│   │
│   ├── models/
│   │   └── schemas.py              # Pydantic schemas
│   │
│   ├── streamlit_app/
│   │   ├── simplified_a2a_client.py # A2A client
│   │   └── simplified_app.py        # Streamlit UI (2 tabs)
│   │
│   └── utils/
│       ├── logger.py               # Logging configuration
│       └── redis_client.py         # Redis client
│
└── data/
    └── policy_processor.db         # SQLite database (auto-created)
```

## Database Schema

### Tables

1. **processing_jobs** - Job metadata and status
2. **policy_documents** - Original PDFs (base64) + metadata
3. **processing_results** - Decision trees + validation results
4. **job_history** - Event tracking and audit log

### Relationships

- One job → One document
- One job → One result
- One job → Many history events

## A2A Protocol

### Agent Card

**Skill ID**: `process_policy`

**Input Parameters**:
- `document_base64` (string, required for new) - Base64-encoded PDF
- `job_id` (string, alternative) - Retrieve existing results
- `use_gpt4` (boolean, default: false) - Use GPT-4 model
- `enable_streaming` (boolean, default: true) - Enable streaming
- `confidence_threshold` (number, default: 0.7) - Validation threshold

**Output**:
- `job_id` (string) - Processing job identifier
- `status` (string) - submitted | processing | completed | failed
- `message` (string) - Human-readable status
- `results` (object, optional) - Complete results if completed

### JSON-RPC Request Example

```json
{
  "jsonrpc": "2.0",
  "method": "sendMessage",
  "params": {
    "message": {
      "messageId": "msg-123",
      "role": "user",
      "parts": [{"kind": "text", "text": "Process policy document"}],
      "taskId": "task-123",
      "contextId": "ctx-123"
    },
    "metadata": {
      "skill_id": "process_policy",
      "parameters": {
        "document_base64": "...",
        "use_gpt4": false,
        "enable_streaming": true,
        "confidence_threshold": 0.7
      }
    }
  },
  "id": "req-123"
}
```

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
```

### Server Configuration

Edit in respective main files:

- **A2A Server Port**: 8001 (in `main_a2a_simplified.py`)
- **Streamlit Port**: 8501 (in `main_streamlit_simplified.py`)
- **Database Path**: `./data/policy_processor.db`

## Streaming Support

Streaming can be controlled by the backend:

```python
# With streaming (real-time progress)
result = client.process_document(
    pdf_bytes=pdf,
    enable_streaming=True
)

# Without streaming (final result only)
result = client.process_document(
    pdf_bytes=pdf,
    enable_streaming=False
)
```

For production deployment, implement streaming event handlers for real-time updates.

## Troubleshooting

### Server Shows Offline

1. Check A2A server is running: `curl http://localhost:8001/health`
2. Check agent card: `curl http://localhost:8001/.well-known/agent-card`
3. Review server logs for errors

### Processing Fails

1. Verify OpenAI API key is set correctly
2. Check Redis is running (if configured)
3. Review A2A server logs for errors
4. Try with smaller document first

### PDF Not in Side-by-Side View

1. Ensure processing completed successfully
2. Check database for `content_base64` in `policy_documents` table
3. Verify job_id is correct

## Production Deployment

### Recommended Enhancements

1. **Authentication** - Add API key or OAuth to A2A server
2. **PostgreSQL** - Replace SQLite with production database
3. **Redis** - Use for task store and caching
4. **Monitoring** - Add Prometheus/Grafana metrics
5. **Logging** - Centralized logging (ELK stack)
6. **Rate Limiting** - Protect against abuse
7. **Load Balancing** - Multiple A2A server instances
8. **WebSocket Events** - Real-time streaming to frontend

### Security Considerations

- Enable HTTPS/TLS for A2A server
- Implement authentication and authorization
- Validate and sanitize all inputs
- Set rate limits per user/API key
- Enable audit logging
- Secure database credentials
- Use environment variables for secrets

## Development

### Running Tests

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=app tests/
```

### Code Formatting

```bash
# Format code
black app/

# Lint code
flake8 app/

# Type checking
mypy app/
```

## License

For authorized use only.

## Support

For issues or questions:

1. Check the logs in both terminals
2. Test A2A server: `curl http://localhost:8001/.well-known/agent-card`
3. Check database: `sqlite3 ./data/policy_processor.db`
4. Review error messages in Streamlit UI

## Version History

### v2.0.0 (Current)
- Simplified A2A architecture with single endpoint
- Fixed "Method not found" error with proper JSON-RPC implementation
- Added PDF storage for side-by-side review
- Clean 2-tab Streamlit interface
- Streaming support for production deployment
- Enhanced database schema with proper relationships

### v1.0.0 (Legacy)
- Initial FastAPI-based implementation
- Multiple A2A endpoints
- Basic Streamlit UI
