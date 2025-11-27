# Quick Start Guide - Stateless Redis Architecture

## Prerequisites

1. **Redis Server**: Must be running on localhost:6379 (or configure in .env)
2. **Python Environment**: Virtual environment with dependencies installed
3. **OpenAI API Key**: Set in .env file

## Installation

```bash
# Install dependencies
pip install -r req.txt

# Verify Redis is running
redis-cli ping
# Should return: PONG
```

## Configuration

Ensure your `.env` file has:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_key_here
OPENAI_MODEL_PRIMARY=gpt-4o-mini
OPENAI_MODEL_SECONDARY=gpt-4o

# API Settings
OPENAI_TIMEOUT=300
OPENAI_MAX_CONCURRENT_REQUESTS=2
OPENAI_PER_REQUEST_TIMEOUT=300

# Chunking Settings
MAX_CHUNK_TOKENS=8000
CHUNK_OVERLAP=600

# Redis Configuration (optional, defaults shown)
# REDIS_HOST=localhost
# REDIS_PORT=6379
# REDIS_DB=0
```

## Running the System

### Step 1: Start Redis (if not already running)

**Windows:**
```bash
# If installed via MSI or zip
redis-server

# Or via Docker
docker run -d -p 6379:6379 redis:latest
```

**Linux/Mac:**
```bash
redis-server

# Or via Docker
docker run -d -p 6379:6379 redis:latest
```

### Step 2: Start A2A Agent Server

```bash
python main_a2a.py
```

Expected output:
```
================================================================================
Policy Document Processor - A2A Server (Stateless)
================================================================================
Server: http://0.0.0.0:8001
Storage: Redis (TTL: 24h)
Architecture: Stateless, container-ready
================================================================================
Redis storage initialized with 24h TTL
PolicyProcessorAgent initialized (stateless, Redis-based)
A2A server created at http://0.0.0.0:8001
```

### Step 3: Start Streamlit UI (in another terminal)

```bash
python main_streamlit.py
```

Expected output:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://xxx.xxx.xxx.xxx:8501
```

### Step 4: Access the Application

Open your browser to: http://localhost:8501

## Quick Test

1. **Upload a PDF**:
   - Click "Choose a PDF policy document"
   - Select a policy document
   - Configure options (GPT-4, confidence threshold)
   - Click "Process Document"

2. **View Results**:
   - Wait for processing to complete (2-5 minutes typical)
   - See summary metrics (Job ID, Policies, Decision Trees)
   - Switch to "Review Decision Trees" tab
   - Enter the Job ID
   - Click "Load Results"

3. **Verify Storage**:
   ```bash
   # Check Redis keys
   redis-cli keys "job:*"

   # Check active jobs
   redis-cli get "agent:active_jobs"

   # Check specific job status
   redis-cli get "job:{your-job-id}:status"
   ```

## Testing Concurrent Requests

1. Open multiple browser tabs to http://localhost:8501
2. Upload different PDFs in each tab simultaneously
3. Click "Process Document" in all tabs
4. Monitor Redis:
   ```bash
   redis-cli get "agent:active_jobs"
   # Should show count > 1
   ```
5. Verify all complete successfully without conflicts

## Troubleshooting

### Redis Connection Error
```
Error: Redis connection refused
```
**Solution**: Start Redis server
```bash
redis-server
```

### A2A Server Offline
```
X A2A Server: Offline
```
**Solution**: Start the A2A server
```bash
python main_a2a.py
```

### Job Not Found
```
No results found for job {id}
```
**Cause**: Redis TTL expired (24h default)
**Solution**: Results are temporary in Redis. Check if job was processed over 24 hours ago.

### Lock Already Acquired
```
Job {id} is already being processed
```
**Cause**: Another instance is processing the same job_id, or previous lock didn't release
**Solution**: Wait 10 minutes for lock to expire, or manually delete lock:
```bash
redis-cli del "job:{job-id}:lock"
```

## Architecture Overview

```
┌─────────────────┐
│  Streamlit UI   │ (Port 8501)
└────────┬────────┘
         │ HTTP/JSON-RPC
         ▼
┌─────────────────┐
│  A2A Agent      │ (Port 8001)
│  (Stateless)    │
└────────┬────────┘
         │
         ├─── Redis ────► Temporary Storage (24h TTL)
         │                - Job status
         │                - Processing locks
         │                - Results
         │
         └─── OpenAI ───► LLM Processing
                         - GPT-4o-mini (default)
                         - GPT-4o (complex)

UI Backend saves to Database:
- PDFs (permanent)
- Results (permanent)
- Job metadata
```

## Redis Data Lifecycle

1. **Job Starts**: Lock acquired, status saved (1h TTL)
2. **Processing**: Status updated periodically
3. **Completion**: Results saved (24h TTL), status updated (24h TTL)
4. **Retrieval**: UI reads results from Redis
5. **Storage**: UI backend saves to database (permanent)
6. **Expiry**: Redis auto-deletes after 24h

## Monitoring

### Check System Health

```bash
# A2A Server health
curl http://localhost:8001/health

# Agent card
curl http://localhost:8001/.well-known/agent-card

# Active jobs
redis-cli get "agent:active_jobs"

# All job keys
redis-cli keys "job:*"
```

### View Logs

Logs are output to console with detailed information:
- `[AGENT]`: Agent operations
- `[UI]`: UI backend operations
- Job IDs in all relevant log messages

## Performance Tips

1. **Concurrent Processing**: System supports 2-3 concurrent OpenAI requests
2. **Large Documents**: Increase `OPENAI_TIMEOUT` for documents >50 pages
3. **Redis Memory**: Monitor Redis memory usage with `redis-cli info memory`
4. **TTL Adjustment**: Modify `result_ttl_hours` in main_a2a.py if needed

## Next Steps

- Review [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) for architecture details
- Run through testing checklist
- Monitor Redis metrics
- Test with production-like documents
- Consider Docker containerization for deployment
