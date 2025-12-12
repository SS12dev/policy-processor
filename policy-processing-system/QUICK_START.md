# Quick Start Guide - Policy Processing System

## ✅ Setup Complete!

Both agent and client modules are configured and ready to run.

---

## 🚀 Running the System

### Prerequisites
- ✅ Python virtual environment: `.venv` (at root)
- ✅ Redis running on localhost:6379
- ✅ All dependencies installed

### Step 1: Start Redis (if not running)

```powershell
# Using Docker
docker run -d -p 6379:6379 redis:latest

# Or check if already running
redis-cli ping
# Should return: PONG
```

---

### Step 2: Start Agent Server (Terminal 1)

```powershell
cd c:\Users\292224\Desktop\policy-processor\policy-processing-system\agent-module
c:\Users\292224\Desktop\policy-processor\.venv\Scripts\python.exe server.py
```

**Expected Output:**
```json
{"timestamp": "2025-12-12T09:15:59.081874", "level": "INFO", "logger": "__main__", "message": "Starting PolicyProcessorAgent"}
INFO:     Started server process [34768]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
```

**Health Check:**
```powershell
curl http://localhost:8001/.well-known/agent-card.json
```

---

### Step 3: Start Client Application (Terminal 2)

```powershell
cd c:\Users\292224\Desktop\policy-processor\policy-processing-system\client-module
c:\Users\292224\Desktop\policy-processor\.venv\Scripts\streamlit run app.py
```

**Expected Output:**
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

---

## 🔍 Verify Everything Works

### Check Agent Health
```powershell
# Get agent card
curl http://localhost:8001/.well-known/agent-card.json

# Response should include:
# {
#   "name": "PolicyProcessorAgent",
#   "version": "4.0.0",
#   "capabilities": {
#     "streaming": true,
#     "push_notifications": false
#   },
#   "skills": [...]
# }
```

### Test Client Connection
1. Open browser to http://localhost:8501
2. You should see "Policy Document Processor" interface
3. Upload a PDF document
4. Watch the agent process it with streaming updates

---

## 📁 Module Structure

```
policy-processing-system/
│
├── agent-module/          # A2A Agent Server ✅ RUNNING
│   ├── server.py          # Main server entry point
│   ├── agent.py           # PolicyProcessorAgent with get_agent()
│   ├── .env               # Agent configuration
│   ├── core/              # LangGraph orchestrator & nodes
│   ├── utils/             # Redis, LLM, logging
│   ├── logs/              # Runtime logs
│   └── metrics/           # Performance metrics
│
└── client-module/         # Streamlit Client ✅ READY
    ├── app.py             # Main Streamlit app
    ├── a2a_client.py      # A2A protocol client
    ├── .env               # Client configuration
    ├── components/        # UI components
    ├── database/          # SQLite operations
    └── data/              # SQLite database
```

---

## 🔧 Configuration

### Agent Module (.env)
- **LLM Provider:** proxy (LiteLLM)
- **Primary Model:** azure/sc-rnd-gpt-4o-mini-01
- **Secondary Model:** azure/sc-rnd-gpt-4o-01
- **Redis:** localhost:6379 (namespace: dev:policy-processor:agent)
- **Server:** 0.0.0.0:8001
- **Streaming:** Enabled
- **Concurrency:** Redis-based locking for safe concurrent processing

### Client Module (.env)
- **Agent URL:** http://localhost:8001
- **Database:** SQLite (./data/policy_processor.db)
- **Max Upload:** 50 MB PDF files
- **Streaming:** Enabled by default

---

## 🎯 How It Works

### Architecture Overview

```
┌─────────────────┐         A2A Protocol          ┌──────────────────┐
│  Streamlit UI   │ ◄────────────────────────────► │  Agent Server    │
│  (Port 8501)    │     HTTP/SSE Streaming        │  (Port 8001)     │
└─────────────────┘                                └──────────────────┘
        │                                                    │
        │ SQLite                                            │ Redis
        ▼                                                    ▼
┌─────────────────┐                                ┌──────────────────┐
│  Local Database │                                │  Redis Storage   │
│  (Results)      │                                │  (Job State)     │
└─────────────────┘                                └──────────────────┘
```

### Request Flow

1. **User uploads PDF** → Streamlit UI (client-module)
2. **Client sends A2A request** → Agent Server (agent-module)
3. **Agent acquires Redis lock** → Prevents duplicate processing
4. **LangGraph workflow executes:**
   - Parse PDF
   - Extract policies
   - Generate decision trees
   - Validate & refine
5. **Agent streams progress** → Client receives real-time updates
6. **Results stored in Redis** → TTL: 24 hours
7. **Client displays results** → Saves to SQLite for history

---

## 🔥 Key Features

### Concurrent Request Handling
- ✅ **Single agent instance** holds workflow definition
- ✅ **Redis-based locking** prevents duplicate processing
- ✅ **Unique job IDs** for state isolation
- ✅ **Horizontal scaling** ready (multiple servers share Redis)

### Production-Ready Architecture
- ✅ **Stateless servers** - All state in Redis
- ✅ **Container-friendly** - No local file dependencies
- ✅ **Streaming support** - Real-time progress updates
- ✅ **Error recovery** - Redis TTL auto-cleanup

---

## 🐛 Troubleshooting

### Redis Not Running?
```powershell
# Start Redis with Docker
docker run -d -p 6379:6379 redis:latest

# Check Redis connection
redis-cli ping
```

### Port Already in Use?
```powershell
# Find process using port 8001
netstat -ano | findstr :8001

# Kill process (replace PID)
taskkill /PID <PID> /F
```

### Agent Won't Start?
```powershell
# Test imports
cd agent-module
c:\Users\292224\Desktop\policy-processor\.venv\Scripts\python.exe -c "from server import app; print('✅ OK')"

# Check Redis connection
c:\Users\292224\Desktop\policy-processor\.venv\Scripts\python.exe -c "from utils.redis_client import get_redis_client; r = get_redis_client(); print(r.ping())"
```

### Client Won't Connect?
1. Check agent is running: `curl http://localhost:8001/.well-known/agent-card.json`
2. Check client `.env`: `AGENT_URL=http://localhost:8001`
3. Check Streamlit logs for connection errors

---

## 📊 Monitoring

### Agent Logs
```powershell
# Watch agent logs (JSON format)
tail -f agent-module/logs/agent.log
```

### Agent Metrics
```powershell
# View performance metrics
curl http://localhost:8001/metrics
```

### Redis Monitor
```powershell
# Watch Redis commands
redis-cli monitor
```

---

## 🚢 Deployment

### Docker (Coming Soon)
```powershell
# Build agent
docker build -t policy-agent ./agent-module

# Run agent
docker run -d -p 8001:8001 --env-file agent-module/.env policy-agent
```

### Environment Variables
- Production: Set `REDIS_NAMESPACE=prod:policy-processor:agent`
- Staging: Set `REDIS_NAMESPACE=staging:policy-processor:agent`
- Use PostgreSQL instead of SQLite for client database

---

## ✨ Success Criteria

You'll know everything is working when:
1. ✅ Agent server starts without errors on port 8001
2. ✅ Agent card is accessible at http://localhost:8001/.well-known/agent-card.json
3. ✅ Streamlit client opens in browser on port 8501
4. ✅ You can upload a PDF and see streaming progress
5. ✅ Results appear with policies and decision trees
6. ✅ Redis shows job state and results (with TTL)

---

## 🎉 You're Ready!

Both modules are now properly structured following industry best practices from the document-analysis-system example. The architecture supports:
- Concurrent request handling
- Horizontal scaling
- Production deployment
- Real-time streaming
- Error recovery

Happy processing! 🚀
