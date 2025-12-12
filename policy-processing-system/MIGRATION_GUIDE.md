# Migration Guide: Modularizing Policy Processing System

This guide explains how to reorganize your policy-processing-system from a unified structure into separate **agent-module** and **client-module** directories.

---

## Overview

### Current Structure

```
policy-processing-system/
├── a2a/
├── core/
├── utils/
├── models/
├── database/
├── streamlit_app/
├── settings.py
├── .env.example
└── requirements.txt
```

### Target Structure

```
policy-processing-system/
├── agent-module/              # A2A Agent Server
│   ├── a2a/
│   ├── core/
│   ├── utils/
│   ├── models/
│   ├── settings.py
│   ├── .env.example
│   ├── requirements.txt
│   └── README.md
│
└── client-module/              # Streamlit Client
    ├── database/
    ├── components/
    ├── app.py
    ├── a2a_client.py
    ├── backend_handler.py
    ├── settings.py
    ├── .env.example
    ├── requirements.txt
    └── README.md
```

---

## Migration Steps

### Step 1: Create Directory Structure

```bash
cd policy-processing-system

# Create module directories
mkdir -p agent-module
mkdir -p client-module
```

### Step 2: Move Agent Files

Move all agent-related files to `agent-module/`:

```bash
# Core agent files
mv a2a/ agent-module/
mv core/ agent-module/
mv utils/ agent-module/
mv models/ agent-module/

# Create empty directories for runtime
mkdir -p agent-module/logs
mkdir -p agent-module/metrics
```

**Files to move:**
- ✅ `a2a/` (entire directory)
- ✅ `core/` (entire directory)
- ✅ `utils/` (entire directory)
- ✅ `models/` (entire directory)

### Step 3: Move Client Files

Move all client-related files to `client-module/`:

```bash
# Client files
mv streamlit_app/database/ client-module/
mv streamlit_app/components/ client-module/
mv streamlit_app/app.py client-module/
mv streamlit_app/a2a_client.py client-module/
mv streamlit_app/backend_handler.py client-module/

# Create runtime directory
mkdir -p client-module/data
```

**Files to move:**
- ✅ `streamlit_app/database/` → `client-module/database/`
- ✅ `streamlit_app/components/` → `client-module/components/`
- ✅ `streamlit_app/app.py` → `client-module/app.py`
- ✅ `streamlit_app/a2a_client.py` → `client-module/a2a_client.py`
- ✅ `streamlit_app/backend_handler.py` → `client-module/backend_handler.py`

### Step 4: Configuration Files

The new configuration files have already been created. Just verify they exist:

**Agent Module:**
```bash
ls agent-module/settings.py        # ✓ Created
ls agent-module/.env.example       # ✓ Created
ls agent-module/requirements.txt   # ✓ Created
ls agent-module/README.md          # ✓ Created
```

**Client Module:**
```bash
ls client-module/settings.py        # ✓ Created
ls client-module/.env.example       # ✓ Created
ls client-module/requirements.txt   # ✓ Created
ls client-module/README.md          # ✓ Created
```

### Step 5: Update Import Statements

#### A. Agent Module Files

**Files to update:**
- `agent-module/a2a/server.py`
- `agent-module/a2a/agent.py`
- All files in `agent-module/core/`

**Change imports from:**
```python
from app.a2a.agent import PolicyProcessorAgent
from app.core.langgraph_orchestrator import LangGraphOrchestrator
from app.utils.llm import get_llm_client
from app.models.schemas import ProcessingRequest
```

**To:**
```python
from a2a.agent import PolicyProcessorAgent
from core.langgraph_orchestrator import LangGraphOrchestrator
from utils.llm import get_llm_client
from models.schemas import ProcessingRequest
```

**Automated replacement (Linux/Mac):**
```bash
cd agent-module
find . -name "*.py" -exec sed -i 's/from app\.//g' {} \;
find . -name "*.py" -exec sed -i 's/import app\.//g' {} \;
```

**Manual replacement (Windows):**
Use your IDE's "Find and Replace" feature:
- Find: `from app.`
- Replace: `from `
- Scope: `agent-module/` directory

#### B. Client Module Files

**Files to update:**
- `client-module/app.py`
- `client-module/a2a_client.py`
- `client-module/backend_handler.py`

**Change imports from:**
```python
from app.streamlit_app.a2a_client import A2AClientSync
from app.database.operations import DatabaseOperations
```

**To:**
```python
from a2a_client import A2AClientSync
from database.operations import DatabaseOperations
```

**Remove path manipulation:**

Find and **remove** these lines in client files:
```python
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

Replace with:
```python
# No path manipulation needed - imports work directly
```

### Step 6: Environment Configuration

#### Agent Module

```bash
cd agent-module
cp .env.example .env
nano .env  # Add your settings
```

Required variables:
```env
OPENAI_API_KEY=sk-...
REDIS_HOST=localhost
REDIS_PORT=6379
```

#### Client Module

```bash
cd client-module
cp .env.example .env
nano .env  # Add your settings
```

Required variables:
```env
AGENT_URL=http://localhost:8001
DATABASE_URL=sqlite:///./data/policy_processor.db
```

### Step 7: Install Dependencies

#### Agent Module

```bash
cd agent-module
pip install -r requirements.txt
```

#### Client Module

```bash
cd client-module
pip install -r requirements.txt
```

### Step 8: Test the Migration

#### Start Agent Server

```bash
cd agent-module
python -m a2a.server
```

Expected output:
```
INFO: Policy Document Processor - A2A Server
INFO: Server: http://0.0.0.0:8001
```

#### Start Client

```bash
cd client-module
streamlit run app.py
```

Expected output:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

### Step 9: Cleanup Old Files

After successful migration and testing:

```bash
# In policy-processing-system/
rm -rf streamlit_app/    # Old client directory
rm settings.py           # Old unified settings
rm .env.example          # Old unified env
rm requirements.txt      # Old unified requirements
rm README.md             # Old unified README
```

---

## Detailed File Mapping

### Agent Module Files

| Source | Destination |
|--------|-------------|
| `a2a/server.py` | `agent-module/a2a/server.py` |
| `a2a/agent.py` | `agent-module/a2a/agent.py` |
| `a2a/redis_storage.py` | `agent-module/a2a/redis_storage.py` |
| `core/*` | `agent-module/core/*` |
| `utils/llm.py` | `agent-module/utils/llm.py` |
| `utils/logger.py` | `agent-module/utils/logger.py` |
| `utils/redis_client.py` | `agent-module/utils/redis_client.py` |
| `models/schemas.py` | `agent-module/models/schemas.py` |

### Client Module Files

| Source | Destination |
|--------|-------------|
| `streamlit_app/app.py` | `client-module/app.py` |
| `streamlit_app/a2a_client.py` | `client-module/a2a_client.py` |
| `streamlit_app/backend_handler.py` | `client-module/backend_handler.py` |
| `streamlit_app/database/*` | `client-module/database/*` |
| `streamlit_app/components/*` | `client-module/components/*` |

---

## Import Update Checklist

### Agent Module

- [ ] `a2a/server.py` - Remove `app.` prefix
- [ ] `a2a/agent.py` - Remove `app.` prefix
- [ ] All `core/*.py` files - Remove `app.` prefix
- [ ] All `utils/*.py` files - Remove `app.` prefix
- [ ] Remove `config.settings` → Use `settings`

### Client Module

- [ ] `app.py` - Remove `app.streamlit_app.` prefix
- [ ] `a2a_client.py` - Remove path manipulation
- [ ] `backend_handler.py` - Update imports
- [ ] Remove `sys.path.insert()` calls

---

## Verification Tests

### Agent Module

```bash
cd agent-module

# Test imports
python -c "from settings import settings; print('✓ Settings OK')"
python -c "from utils.llm import get_llm_client; print('✓ LLM client OK')"
python -c "from a2a.agent import PolicyProcessorAgent; print('✓ Agent OK')"

# Start server (should not error)
python -m a2a.server
```

### Client Module

```bash
cd client-module

# Test imports
python -c "from settings import settings; print('✓ Settings OK')"
python -c "from database.operations import DatabaseOperations; print('✓ Database OK')"
python -c "from a2a_client import A2AClientSync; print('✓ A2A client OK')"

# Start app (should not error)
streamlit run app.py
```

---

## Common Issues & Solutions

### Issue 1: Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'app.utils.llm'
```

**Solution:**
Update imports to remove `app.` prefix:
```python
# Before
from app.utils.llm import get_llm_client

# After
from utils.llm import get_llm_client
```

### Issue 2: Settings Not Found

**Error:**
```
ImportError: cannot import name 'settings' from 'config.settings'
```

**Solution:**
```python
# Before
from config.settings import settings

# After
from settings import settings
```

### Issue 3: Path Issues

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'logs/agent.log'
```

**Solution:**
Create directories:
```bash
mkdir -p agent-module/logs
mkdir -p agent-module/metrics
mkdir -p client-module/data
```

### Issue 4: A2A Connection Failed

**Error:**
```
Connection refused to http://localhost:8001
```

**Solution:**
1. Ensure agent is running: `cd agent-module && python -m a2a.server`
2. Check `AGENT_URL` in client `.env`
3. Verify port 8001 is not blocked

---

## Rollback Plan

If migration fails, restore from backup:

```bash
# Undo changes
git checkout .
git clean -fd

# Or restore from backup
cp -r /path/to/backup/* .
```

---

## Post-Migration Checklist

- [ ] Both modules start without errors
- [ ] Agent responds to health checks
- [ ] Client connects to agent successfully
- [ ] Document upload works
- [ ] Processing completes successfully
- [ ] Results display correctly
- [ ] Database saves records
- [ ] Logs are created
- [ ] Metrics are tracked
- [ ] All imports resolved

---

## Support

If you encounter issues during migration:

1. Check the console output for specific errors
2. Verify all imports are updated
3. Ensure `.env` files are configured
4. Test each module independently
5. Check logs in `agent-module/logs/` and `client-module/data/`

For additional help, open an issue with:
- Error messages
- Steps to reproduce
- Environment details (OS, Python version)
