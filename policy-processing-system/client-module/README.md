# Policy Document Processor - Client Module

**Streamlit-based web interface for policy document analysis.**

---

## Overview

The client module provides an interactive web UI for uploading policy documents, tracking processing progress, viewing results, and visualizing decision trees. It communicates with the agent module via the A2A protocol.

### Key Features

- ✅ **Interactive UI**: Streamlit-based web interface
- ✅ **A2A Client**: Connects to agent via A2A protocol
- ✅ **Real-time Progress**: Streaming status updates
- ✅ **Tree Visualization**: Interactive decision tree display
- ✅ **History Tracking**: SQLite database for results
- ✅ **Export Options**: JSON and CSV downloads

---

## Quick Start

### 1. Installation

```bash
cd client-module
pip install -r requirements.txt
```

### 2. Configuration

```bash
cp .env.example .env
nano .env  # Configure agent URL
```

**Required Settings:**
```env
AGENT_URL=http://localhost:8001
```

### 3. Ensure Agent is Running

The client needs the agent module running:

```bash
# In another terminal
cd ../agent-module
python -m a2a.server
```

### 4. Start Client

```bash
streamlit run app.py
```

UI opens at `http://localhost:8501`

---

## Directory Structure

```
client-module/
├── database/                  # Database Layer
│   ├── __init__.py
│   ├── models.py              # SQLAlchemy models
│   └── operations.py          # CRUD operations
│
├── components/                # UI Components
│   ├── __init__.py
│   ├── metrics_dashboard.py  # Metrics display
│   ├── tree_visualizer.py    # Tree visualization
│   └── tree_renderers/       # Tree rendering modules
│       ├── __init__.py
│       ├── decision_tree_renderer.py
│       └── hierarchy_renderer.py
│
├── data/                      # Runtime data (created automatically)
│   └── policy_processor.db   # SQLite database
│
├── app.py                     # Main Streamlit application
├── a2a_client.py              # A2A client wrapper
├── backend_handler.py         # Response processing
├── settings.py                # Client configuration
├── .env.example               # Configuration template
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Usage

### Uploading Documents

1. **Open Browser**: Navigate to `http://localhost:8501`
2. **Upload PDF**: Click "Choose a PDF file" or drag & drop
3. **Configure Options**:
   - **Use GPT-4**: Enable for complex documents (slower, more accurate)
   - **Confidence Threshold**: 0.0-1.0 (default: 0.7)
   - **Enable Streaming**: See real-time progress
4. **Process**: Click "Process Document"
5. **View Results**: See policies, decision trees, and validation

### Viewing History

1. Switch to **"Review Decision Trees"** tab
2. Select a previously processed document
3. View saved results and decision trees

### Exporting Results

- **Download JSON**: Full structured results
- **Download CSV**: Flattened summary data

---

## Configuration

### Environment Variables

All settings are configured via `.env` file.

**Agent Connection:**

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_URL` | `http://localhost:8001` | A2A agent endpoint |
| `AGENT_TIMEOUT` | `300` | Request timeout (seconds) |
| `AGENT_PREFER_STREAMING` | `true` | Use streaming responses |

**Upload Settings:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_UPLOAD_SIZE_MB` | `50` | Max file size |
| `ALLOWED_FILE_TYPES` | `pdf` | Allowed types |

**UI Settings:**

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_TITLE` | `Policy Document Processor` | Application title |
| `PAGE_LAYOUT` | `wide` | Layout mode |
| `SIDEBAR_STATE` | `expanded` | Sidebar default |

**Database:**

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./data/policy_processor.db` | Database connection |

See [.env.example](.env.example) for all options.

---

## Architecture

### Components

```
┌─────────────────────┐
│   Streamlit App     │  ← User Interface (app.py)
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│   A2A Client        │  ← Protocol Communication (a2a_client.py)
└──────────┬──────────┘
           │ A2A Protocol
           ↓
┌─────────────────────┐
│   Agent Module      │  ← Processing Backend
│   (Port 8001)       │
└─────────────────────┘

Local Storage:
┌─────────────────────┐
│  SQLite Database    │  ← Results History
└─────────────────────┘
```

### Data Flow

1. **Upload**: User uploads PDF via Streamlit
2. **Encode**: PDF converted to base64
3. **Send**: A2A client sends to agent
4. **Stream**: Real-time status updates
5. **Receive**: Artifact with results
6. **Store**: Save to SQLite database
7. **Display**: Render trees and results

---

## Features

### 1. Document Upload

- Drag & drop or file picker
- File size validation
- Type checking (PDF only)
- Base64 encoding
- Progress indication

### 2. Real-time Processing

- Status updates during processing
- Progress bar with stages
- Live log streaming
- Error handling with retry

### 3. Decision Tree Visualization

- Hierarchical tree rendering
- Interactive node expansion
- AND/OR logic grouping
- Eligibility question display
- Export to various formats

### 4. Results Management

- SQLite storage
- History browsing
- Re-analysis capability
- Comparison tools
- Bulk export

### 5. Metrics Dashboard

- Processing statistics
- Success/failure rates
- Average processing time
- Document type distribution

---

## Database Schema

### Documents Table

```sql
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    filename TEXT NOT NULL,
    file_hash TEXT UNIQUE,
    file_size_bytes INTEGER,
    upload_timestamp DATETIME,
    pdf_content BLOB
);
```

### Processing Jobs Table

```sql
CREATE TABLE processing_jobs (
    id INTEGER PRIMARY KEY,
    document_id INTEGER,
    job_id TEXT UNIQUE,
    status TEXT,
    request_timestamp DATETIME,
    completion_timestamp DATETIME,
    use_gpt4 BOOLEAN,
    confidence_threshold REAL,
    FOREIGN KEY (document_id) REFERENCES documents(id)
);
```

### Results Table

```sql
CREATE TABLE processing_results (
    id INTEGER PRIMARY KEY,
    job_id TEXT UNIQUE,
    policy_hierarchy JSON,
    decision_trees JSON,
    validation_result JSON,
    processing_stats JSON,
    FOREIGN KEY (job_id) REFERENCES processing_jobs(job_id)
);
```

---

## Development

### Running in Development Mode

```bash
# Enable debug mode
export SHOW_DEBUG_INFO=true

# Run with auto-reload
streamlit run app.py --server.runOnSave true
```

### Adding Custom Components

1. **Create component** in `components/`:

```python
# components/my_component.py
import streamlit as st

def render_my_component(data):
    st.header("My Component")
    # Rendering logic
```

2. **Import in app.py**:

```python
from components.my_component import render_my_component

# Use in app
render_my_component(results)
```

### Styling

Streamlit uses markdown and CSS. Custom styles in app.py:

```python
st.markdown("""
<style>
.my-class {
    color: #007bff;
}
</style>
""", unsafe_allow_html=True)
```

---

## Troubleshooting

### Agent Connection Failed

```bash
# Check agent is running
curl http://localhost:8001/health

# Check network connectivity
ping localhost

# Verify agent URL in .env
echo $AGENT_URL
```

### Database Errors

```bash
# Reset database
rm data/policy_processor.db

# Recreate schema (automatic on next run)
streamlit run app.py
```

### Upload Failures

- Check file size < `MAX_UPLOAD_SIZE_MB`
- Ensure file is valid PDF
- Verify agent is responsive

### Display Issues

- Clear browser cache
- Refresh page (Ctrl+F5)
- Check console for errors (F12)

---

## API Reference

### A2A Client

```python
from a2a_client import A2AClientSync
from settings import settings

# Create client
client = A2AClientSync(
    base_url=settings.agent_url,
    timeout=settings.agent_timeout
)

# Process document
results = client.process_document(
    pdf_bytes=pdf_data,
    use_gpt4=False,
    confidence_threshold=0.7
)
```

### Database Operations

```python
from database.operations import DatabaseOperations
from settings import settings

# Initialize
db = DatabaseOperations(database_url=settings.database_url)

# Store document
doc_id = db.store_document(
    filename="policy.pdf",
    pdf_bytes=pdf_data
)

# Get results
results = db.get_processing_results(job_id="uuid")
```

---

## Deployment

### Production Deployment

```bash
# Set production config
export AGENT_URL=https://agent.example.com
export DATABASE_URL=postgresql://...

# Run with optimizations
streamlit run app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t policy-client .
docker run -p 8501:8501 \
  -e AGENT_URL=http://agent:8001 \
  policy-client
```

---

## Security

### Best Practices

- Never commit `.env` files
- Use environment variables for secrets
- Validate all user inputs
- Sanitize file uploads
- Implement rate limiting
- Use HTTPS in production
- Enable CORS restrictions

### Environment Security

```env
# Production .env
AGENT_URL=https://secure-agent.example.com
DATABASE_URL=postgresql://encrypted-connection
SHOW_DEBUG_INFO=false
```

---

## License

MIT License - See LICENSE file

---

## Support

- **Issues**: https://github.com/your-repo/issues
- **Documentation**: https://docs.example.com
- **Email**: support@example.com
