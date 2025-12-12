# PDF Document Analysis System

A complete, production-ready document analysis system built with:
- **Agent Module**: LangGraph-based PDF analysis agent with A2A protocol
- **Client Module**: Streamlit UI with SQLite storage and A2A client

## Features

### Agent Module
- ✅ **Multi-Node LangGraph Pipeline**:
  - PDF Validation
  - PDF Text Extraction
  - Heading Extraction
  - Keyword Extraction (Basic NLP + LLM hybrid)
  - LLM-powered Document Analysis
  - Response Formatting
- ✅ **Conditional Edges & Retry Logic**
- ✅ **Streaming Support** (control from client)
- ✅ **A2A Protocol Compliant**
- ✅ **Comprehensive Logging & Metrics**
- ✅ **Rate Limiting**
- ✅ **Fully Configurable via .env**

### Client Module
- ✅ **Streamlit Web UI**
- ✅ **PDF Upload with Validation**
- ✅ **Streaming/Non-Streaming Toggle**
- ✅ **Real-time Progress Updates**
- ✅ **SQLite Database** (stores PDFs as BLOBs + results)
- ✅ **Analysis History**
- ✅ **Export Results** (JSON/CSV)
- ✅ **Fully Configurable via .env**

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Module                            │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐               │
│  │ Streamlit  │  │  A2A       │  │  SQLite    │               │
│  │    UI      │──│  Client    │──│  Database  │               │
│  └────────────┘  └────────────┘  └────────────┘               │
└───────────────────────────┬─────────────────────────────────────┘
                            │ A2A Protocol (HTTP/JSON-RPC)
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                        Agent Module                             │
│  ┌────────────┐  ┌─────────────────────────────────┐           │
│  │  A2A       │  │     LangGraph Agent             │           │
│  │  Server    │──│  ┌──────────┐  ┌──────────┐    │           │
│  └────────────┘  │  │Validator │──│  Parser  │    │           │
│                  │  └──────────┘  └──────────┘    │           │
│                  │        │             │          │           │
│                  │  ┌──────────┐  ┌──────────┐    │           │
│                  │  │Heading   │  │Keyword   │    │           │
│                  │  │Extractor │  │Extractor │    │           │
│                  │  └──────────┘  └──────────┘    │           │
│                  │        │             │          │           │
│                  │  ┌──────────┐  ┌──────────┐    │           │
│                  │  │   LLM    │──│Response  │    │           │
│                  │  │Analyzer  │  │Formatter │    │           │
│                  │  └──────────┘  └──────────┘    │           │
│                  └─────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites
- Python 3.10 or higher
- OpenAI API key

### Agent Module Setup

1. **Navigate to agent module**:
```bash
cd agent-module
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment**:
```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY and other settings
```

5. **Create necessary directories**:
```bash
mkdir -p logs metrics
```

### Client Module Setup

1. **Navigate to client module**:
```bash
cd ../client-module
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment**:
```bash
cp .env.example .env
# Edit .env to match your setup (agent URL, preferences, etc.)
```

5. **Create necessary directories**:
```bash
mkdir -p data logs
```

## Configuration

### Agent Module (.env)

Key configuration options:

```env
# Agent Identity
AGENT_NAME=PDF Document Analyzer
AGENT_VERSION=1.0.0
AGENT_URL=http://localhost:8000

# LLM Configuration
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.3

# Processing
MAX_KEYWORDS=15
MAX_HEADINGS=30
ENABLE_LLM_ANALYSIS=true

# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
MAX_REQUEST_SIZE_MB=100

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=30

# Streaming
ENABLE_STREAMING=true
```

### Client Module (.env)

Key configuration options:

```env
# Agent Connection
AGENT_URL=http://localhost:8000
AGENT_TIMEOUT_SECONDS=300

# Upload
MAX_FILE_SIZE_MB=50
ALLOWED_FILE_TYPES=pdf

# Streaming
PREFER_STREAMING=true

# UI
APP_TITLE=PDF Document Analysis Client
APP_PORT=8501

# Database
DB_PATH=./data/client.db
```

## Running the System

### 1. Start the Agent Server

```bash
cd agent-module
source venv/bin/activate  # On Windows: venv\Scripts\activate
python server.py
```

The agent will start on `http://localhost:8000`

Check health: `http://localhost:8000/health`
View metrics: `http://localhost:8000/metrics`
Agent card: `http://localhost:8000/.well-known/agent-card.json`

### 2. Start the Client UI

In a new terminal:

```bash
cd client-module
source venv/bin/activate  # On Windows: venv\Scripts\activate
streamlit run app.py
```

The UI will open in your browser at `http://localhost:8501`

## Usage

### Upload and Analyze Documents

1. **Open the Streamlit UI** (`http://localhost:8501`)
2. **Navigate to "Upload & Analyze" tab**
3. **Configure settings**:
   - Toggle streaming on/off
   - Adjust max keywords/headings if needed
4. **Upload a PDF file** (drag & drop or browse)
5. **Click "Analyze Document"**
6. **Watch real-time progress** (if streaming enabled)
7. **View results**:
   - Document info (pages, words, size)
   - Extracted headings
   - Keywords
   - AI-powered analysis (summary, topics, insights)
8. **Download results** as JSON or CSV

### View History

1. **Navigate to "History" tab**
2. **Browse previous analyses**
3. **Filter by date, filename, or status**
4. **View detailed results**
5. **Re-analyze documents**
6. **Export history**

### Monitor Agent Performance

1. **Navigate to "Metrics" tab** (in client UI)
2. **Or visit** `http://localhost:8000/metrics`
3. **View**:
   - Total requests processed
   - Success/failure rates
   - Processing time averages
   - LLM usage and costs
   - Error statistics
   - Per-node execution times

## File Upload Size Limits

### Configuration

**Agent Side** (controls max request size):
```env
MAX_REQUEST_SIZE_MB=100
```

**Client Side** (controls upload validation):
```env
MAX_FILE_SIZE_MB=50
```

### Best Practices

- For files **< 10 MB**: Upload directly (uses FileWithBytes)
- For files **> 50 MB**: Consider:
  1. Increasing limits in both .env files
  2. Using compression before upload
  3. Splitting into smaller documents

### Technical Notes

- PDFs are stored as **BLOBs** in SQLite
- Base64 encoding adds ~33% overhead
- Streaming helps with large files
- Files are deduplicated by checksum

## Streaming vs Non-Streaming

### How It Works

The **agent is designed for streaming** - it yields updates as it processes. The **client controls** whether to use streaming or polling.

#### Streaming Mode (Recommended)

```env
# Client .env
PREFER_STREAMING=true
```

**Advantages**:
- Real-time progress updates
- See results as they're generated
- Better UX for long documents
- Lower perceived latency

#### Non-Streaming Mode (Polling)

```env
# Client .env
PREFER_STREAMING=false
POLL_INTERVAL_SECONDS=1.0
MAX_POLL_ATTEMPTS=180
```

**Advantages**:
- Simpler error handling
- Works with any HTTP proxy
- Easier debugging

### Controlling from UI

The Streamlit app has a toggle switch:
- **ON**: Real-time streaming updates
- **OFF**: Wait for complete results

## Database Schema

### Documents Table
```sql
- id: INTEGER PRIMARY KEY
- filename: TEXT
- file_size_bytes: INTEGER
- file_data: BLOB (PDF content)
- upload_timestamp: TEXT
- mime_type: TEXT
- checksum: TEXT (for deduplication)
```

### Requests Table
```sql
- id: INTEGER PRIMARY KEY
- document_id: INTEGER (FK)
- context_id: TEXT
- task_id: TEXT
- request_timestamp: TEXT
- streaming_enabled: BOOLEAN
- status: TEXT
```

### Responses Table
```sql
- id: INTEGER PRIMARY KEY
- request_id: INTEGER (FK)
- response_timestamp: TEXT
- processing_time_seconds: REAL
- status: TEXT
- response_data: TEXT (JSON)
- error_message: TEXT
```

### Extracted Results Table
```sql
- id: INTEGER PRIMARY KEY
- response_id: INTEGER (FK)
- page_count: INTEGER
- word_count: INTEGER
- heading_count: INTEGER
- keyword_count: INTEGER
- headings: TEXT (JSON)
- keywords: TEXT (JSON)
- summary: TEXT
- document_type: TEXT
```

## LangGraph Workflow

### Nodes

1. **validate_pdf**: Check file format, size, integrity
2. **parse_pdf**: Extract text using PyPDF2
3. **extract_headings**: Find document structure
4. **extract_keywords**: Hybrid basic NLP + LLM
5. **analyze_document**: Deep LLM analysis
6. **format_response**: Structure final output

### Edges & Conditions

- **START → validate_pdf**
- **validate_pdf → parse_pdf** (if valid)
- **parse_pdf → retry** (if failed and retries left)
- **parse_pdf → extract_headings** (if success)
- **extract_headings → extract_keywords**
- **extract_keywords → analyze_document** (if LLM enabled)
- **extract_keywords → format_response** (if LLM disabled)
- **analyze_document → format_response**
- **format_response → END**

### State Management

LangGraph's `MemorySaver` checkpointer preserves:
- Conversation context (via `context_id`)
- Processing state (for retry logic)
- Intermediate results (between nodes)

## Logging

### Agent Logs

Location: `agent-module/logs/agent.log`

Format: JSON (configurable to text)

Includes:
- Node execution times
- LLM API calls and tokens
- Errors with stack traces
- Request/response details

### Client Logs

Location: `client-module/logs/client.log`

Includes:
- UI interactions
- API requests
- Database operations
- Errors

### Log Rotation

- **Size**: 10MB per file (configurable)
- **Retention**: 30 days (configurable)
- **Format**: JSON or text

## Metrics

### Available Metrics

**Requests**:
- Total, successful, failed counts
- By status breakdown

**Processing**:
- Total documents processed
- Average processing time
- Per-node execution times

**LLM**:
- API call count
- Token usage
- Cost estimates
- By model breakdown

**Errors**:
- Total error count
- By error type
- Last occurrence timestamps

### Accessing Metrics

**Via API**:
```bash
curl http://localhost:8000/metrics
```

**Via UI**:
Navigate to Metrics tab in Streamlit app

## Error Handling

### Validation Errors

- File type checking
- File size limits
- PDF corruption detection
- Automatic user feedback

### Processing Errors

- **Retry logic**: Configurable retries with exponential backoff
- **Graceful degradation**: Partial results if some nodes fail
- **Detailed error messages**: Logged and returned to client

### Connection Errors

- **Auto-retry**: Connection failures retry automatically
- **Timeout handling**: Configurable timeouts
- **Fallback**: Switch to polling if streaming fails

## Troubleshooting

### Agent Won't Start

1. Check `.env` file exists and has valid `OPENAI_API_KEY`
2. Verify port 8000 is available: `lsof -i :8000`
3. Check logs: `tail -f agent-module/logs/agent.log`

### Client Can't Connect

1. Verify agent is running: `curl http://localhost:8000/health`
2. Check `AGENT_URL` in client `.env`
3. Check firewall settings

### PDF Upload Fails

1. Check file size: Must be < `MAX_FILE_SIZE_MB`
2. Verify PDF is not corrupted: Open in PDF reader
3. Check logs for specific error

### Processing Takes Too Long

1. Increase `AGENT_TIMEOUT_SECONDS` in client `.env`
2. Disable LLM analysis for faster processing:
   ```env
   ENABLE_LLM_ANALYSIS=false
   ```
3. Reduce document size or split into parts

### Out of Memory

1. Reduce `MAX_REQUEST_SIZE_MB`
2. Process smaller documents
3. Increase system memory
4. Consider using FileWithUri for very large files

## Development

### Project Structure

```
document-analysis-system/
├── agent-module/
│   ├── .env.example
│   ├── settings.py
│   ├── server.py
│   ├── agent.py
│   ├── requirements.txt
│   ├── nodes/
│   │   ├── pdf_parser.py
│   │   ├── heading_extractor.py
│   │   ├── keyword_extractor.py
│   │   ├── llm_analyzer.py
│   │   └── response_formatter.py
│   ├── utils/
│   │   ├── llm.py
│   │   ├── logger.py
│   │   └── metrics.py
│   ├── logs/
│   └── metrics/
├── client-module/
│   ├── .env.example
│   ├── settings.py
│   ├── app.py (Streamlit)
│   ├── a2a_client.py
│   ├── database.py
│   ├── requirements.txt
│   ├── data/
│   └── logs/
└── README.md
```

### Adding New Features

#### Add a New Node to Agent

1. Create node file in `agent-module/nodes/`
2. Implement async function that takes state and returns updated state
3. Add node to graph in `agent.py`:
   ```python
   workflow.add_node("my_node", my_node_function)
   workflow.add_edge("previous_node", "my_node")
   ```
4. Update state schema in `agent.py`

#### Add New LLM Provider

1. Edit `agent-module/utils/llm.py`
2. Add provider-specific client initialization
3. Update `get_langchain_model()` method
4. Add provider settings to `settings.py`

#### Customize UI

1. Edit `client-module/app.py`
2. Streamlit components are modular
3. Add new tabs with `st.tabs()`
4. Add new visualizations with Plotly

### Testing

#### Test Agent

```bash
cd agent-module
python -m pytest tests/  # (create tests as needed)
```

Or manual test:
```bash
curl -X POST http://localhost:8000/rpc \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"message/send","id":1,"params":{...}}'
```

#### Test Client

```bash
cd client-module
# Run Streamlit and test via UI
streamlit run app.py
```

## Production Deployment

### Recommended Setup

**Agent**:
- Deploy behind reverse proxy (nginx)
- Use Gunicorn with multiple workers
- Enable HTTPS
- Set up monitoring (Prometheus, Grafana)
- Configure log aggregation

**Client**:
- Deploy Streamlit on separate server
- Use authentication (Streamlit auth or SSO)
- Regular database backups
- CDN for static assets

### Docker Deployment (Optional)

Create `Dockerfile` for each module:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "server.py"]  # or ["streamlit", "run", "app.py"]
```

### Security Considerations

1. **Never commit .env files**
2. **Rotate API keys regularly**
3. **Use environment-specific configs**
4. **Enable rate limiting in production**
5. **Sanitize file uploads**
6. **Use HTTPS in production**
7. **Implement authentication**

## License

This project is provided for educational and research purposes.

## Support

For issues or questions:
1. Check this README
2. Review logs in `logs/` directories
3. Check agent `/health` and `/metrics` endpoints
4. Verify all `.env` settings

## Acknowledgments

- Built with [A2A SDK](https://github.com/a2aproject/a2a-python)
- Uses [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- UI powered by [Streamlit](https://streamlit.io)
- LLM integration via [OpenAI](https://openai.com)

---

**Version**: 1.0.0
**Last Updated**: December 2025
**Python**: 3.10+
**A2A SDK**: 0.3.20+
