# Quick Start Guide

## Prerequisites

- Python 3.10+
- OpenAI API key
- 15 minutes

## Setup (5 minutes)

### 1. Agent Module

```bash
cd agent-module

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
nano .env  # Add your OPENAI_API_KEY

# Create directories
mkdir -p logs metrics
```

### 2. Client Module

```bash
cd ../client-module

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Default settings should work, but you can customize

# Create directories
mkdir -p data logs
```

## Run (2 minutes)

### Terminal 1: Start Agent

```bash
cd agent-module
source venv/bin/activate
python server.py
```

Wait for: `INFO: Uvicorn running on http://0.0.0.0:8000`

### Terminal 2: Start Client

```bash
cd client-module
source venv/bin/activate
streamlit run app.py
```

Browser will open automatically at `http://localhost:8501`

## Use (2 minutes)

1. **Upload PDF**: Drag & drop a PDF file
2. **Configure**: Toggle streaming ON (recommended)
3. **Analyze**: Click "🚀 Analyze Document"
4. **Watch**: See real-time progress updates
5. **Review**: View headings, keywords, and AI analysis
6. **Download**: Export results as JSON or CSV

## Verify It Works

### Test Agent Health

```bash
curl http://localhost:8000/health
```

Should return:
```json
{"status": "healthy", "agent": "PDF Document Analyzer"}
```

### Test Agent Card

```bash
curl http://localhost:8000/.well-known/agent-card.json
```

Should return agent capabilities.

### Upload Test PDF

1. Open UI: http://localhost:8501
2. Upload any PDF (try a research paper or report)
3. Click "Analyze Document"
4. Should complete in 10-30 seconds

## Example Output

For a 10-page research paper, you'll get:

- ✅ **Document Info**: Pages, words, size
- ✅ **Headings**: 15-30 structural headings
- ✅ **Keywords**: 10-15 key terms
- ✅ **AI Analysis**:
  - Summary (2-3 sentences)
  - Document type (e.g., "research paper")
  - Main topics
  - Key insights
  - Complexity level

## Common Issues

### "Module not found" Error

```bash
# Make sure you're in the virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### "OpenAI API key not set"

```bash
# Edit .env file in agent-module
nano agent-module/.env
# Add: OPENAI_API_KEY=sk-...
```

### Port Already in Use

```bash
# Agent (port 8000)
lsof -i :8000  # Find process
kill <PID>     # Kill it

# Client (port 8501)
lsof -i :8501
kill <PID>
```

### Agent Connection Failed

1. Verify agent is running: `curl http://localhost:8000/health`
2. Check `AGENT_URL` in `client-module/.env`
3. Check firewall settings

## Next Steps

- ✅ Try different PDFs (reports, papers, books)
- ✅ Toggle streaming off to see polling mode
- ✅ Check History tab for past analyses
- ✅ View Stats tab for metrics
- ✅ Export results as JSON/CSV
- ✅ Check logs in `logs/` directories
- ✅ View metrics at `http://localhost:8000/metrics`

## Configuration Tips

### For Faster Processing

```env
# In agent-module/.env
ENABLE_LLM_ANALYSIS=false  # Skip AI analysis
OPENAI_MODEL=gpt-3.5-turbo  # Use faster model
```

### For Larger Files

```env
# In agent-module/.env
MAX_REQUEST_SIZE_MB=200  # Increase limit

# In client-module/.env
MAX_FILE_SIZE_MB=150  # Increase limit
AGENT_TIMEOUT_SECONDS=600  # Increase timeout
```

### For Cost Savings

```env
# In agent-module/.env
OPENAI_MODEL=gpt-4o-mini  # Cheapest model
OPENAI_TEMPERATURE=0.1  # More deterministic
MAX_KEYWORDS=10  # Reduce LLM calls
```

## Full Documentation

See [README.md](README.md) for:
- Complete configuration reference
- Architecture details
- Database schema
- Production deployment
- Development guide
- Troubleshooting

## Support

Issues? Check:
1. This guide
2. [README.md](README.md)
3. Logs in `logs/` directories
4. `/health` and `/metrics` endpoints

---

**Time to first analysis**: ~7 minutes from clone to results!
