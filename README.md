# Policy Document Processor

Advanced LangGraph-powered system for processing policy documents into structured decision trees with conditional routing, interactive visualization, and comprehensive editing capabilities.

## Key Features

### Decision Trees with Conditional Routing
- Explicit routing rules for every question and answer path
- Support for yes/no, multiple-choice, and numeric range questions
- AND/OR logic grouping for complex policy conditions
- Complete path coverage validation
- Multiple outcome types: approved, denied, review, documentation required

### Interactive Tile-Based Visualization
- Vertical tree layout with indentation showing hierarchy
- Color-coded nodes by type (questions, decisions, outcomes)
- Collapsible branches for easy navigation
- Answer labels showing conditional flow
- Path highlighting and tracing
- Side-by-side PDF comparison

### Full Editing Capabilities
- Visual tree editor with inline editing
- Add, remove, and reorder nodes
- Configure routing rules through GUI
- Path validation and completeness checking
- Bulk operations and import/export

### Intelligent Processing
- Semantic-aware document chunking
- Hierarchical policy extraction
- Confidence-based validation
- Automatic retry for low-confidence sections
- Source reference tracking

## Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API key
- Redis server
- Tesseract OCR (optional, for scanned PDFs)

### Installation

```bash
# Clone and setup
git clone <repository-url>
cd policy-processor
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with:
# - OPENAI_API_KEY=your-key
# - REDIS_HOST=localhost
# - REDIS_PORT=6379

# Start Redis
redis-server

# Run migrations
python migrate_add_policy_name.py
```

### Running the Application

**Terminal 1 - Start A2A Server:**
```bash
python main_a2a.py
```
Server available at http://localhost:8001

**Terminal 2 - Start Streamlit UI:**
```bash
streamlit run app/streamlit_app/app.py
```
UI available at http://localhost:8501

## Usage Guide

### Processing a Policy Document

1. **Upload & Process Tab**
   - Enter unique policy name
   - Upload PDF document
   - Configure options:
     - Use GPT-4 for complex sections
     - Set confidence threshold (default: 0.7)
   - Click "Process Document"
   - Monitor real-time progress

2. **View Generated Trees**
   - Navigate to "View Policy" tab
   - Select policy from dropdown
   - Choose visualization mode:
     - Interactive Tree Explorer (tile-based)
     - Side-by-Side (PDF + Tree)
     - Summary (traditional text)

3. **Edit and Refine**
   - Navigate to "Review & Edit" tab
   - Load policy
   - Enable editing mode
   - Modify structure:
     - Edit questions and routing
     - Add/remove nodes
     - Configure conditional paths
     - Set outcome types
   - Save changes

## Architecture

### System Components

```
Streamlit UI
    ├── Upload & Process (Tab 1)
    ├── View Policy (Tab 2)
    └── Review & Edit (Tab 3)
         │
         ▼ (A2A Protocol)
A2A Server (FastAPI)
    └── PolicyProcessorAgent
         └── LangGraph Orchestrator
              ├── parse_pdf_node
              ├── analyze_document_node
              ├── chunk_document_node (semantic)
              ├── extract_policy_node
              ├── generate_trees_node (with routing)
              ├── validate_results_node
              └── retry_logic_node
                   │
                   ▼
Database (SQLite)
    ├── ProcessingJobs
    ├── PolicyDocuments
    └── ProcessingResults
```

### Decision Tree Structure

Enhanced node types with routing:

```python
{
  "node_id": "age_check",
  "node_type": "question",
  "question": {
    "question_text": "Are you 18 or older?",
    "question_type": "yes_no"
  },
  "children": {
    "yes": { /* next question */ },
    "no": { /* denial outcome */ }
  },
  "routing_rules": [{
    "answer_value": "yes",
    "comparison": "equals",
    "next_node_id": "insurance_check"
  }],
  "confidence_score": 0.95
}
```

## Core Components

### Decision Tree Generation (`app/core/decision_tree_generator.py`)
- Generates trees with explicit routing
- Validates path completeness
- Handles hierarchical policies
- Supports AND/OR logic

### Tree Validation (`app/core/tree_validator.py`)
- Validates routing completeness
- Checks node reachability
- Identifies incomplete paths
- Validates logic groups

### Tile Visualizer (`app/streamlit_app/components/tree_visualizer.py`)
- Renders vertical tile-based trees
- Color-codes node types
- Shows answer labels
- Supports path highlighting

### Enhanced Prompts (`app/core/tree_generation_prompts.py`)
- Structured prompts for routing generation
- Validation instructions
- Complete path coverage requirements

## Configuration

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your-openai-api-key

# Optional
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Model Configuration
OPENAI_MODEL_PRIMARY=gpt-4o-mini      # For extraction
OPENAI_MODEL_SECONDARY=gpt-4o         # For tree generation
MAX_CONCURRENT_REQUESTS=5
PER_REQUEST_TIMEOUT=300
```

### Processing Options
- `use_gpt4`: Use GPT-4 for complex sections
- `confidence_threshold`: Minimum confidence (0-1)
- `enable_streaming`: Real-time updates
- `max_depth`: Maximum hierarchy depth

## API Reference

### Tree Generation
```python
from app.core.decision_tree_generator import DecisionTreeGenerator

generator = DecisionTreeGenerator(use_gpt4=True)
tree = await generator.generate_tree_for_policy(policy)
```

### Tree Validation
```python
from app.core.tree_validator import TreeValidator

validator = TreeValidator()
is_valid, unreachable, incomplete = validator.validate_tree(tree)
paths = validator.get_all_paths(tree)
```

### Visualization
```python
from app.streamlit_app.components import render_tile_tree_view

render_tile_tree_view(tree_data)
```

## Project Structure

```
policy-processor/
├── app/
│   ├── core/
│   │   ├── decision_tree_generator.py    # Tree generation with routing
│   │   ├── tree_validator.py              # Path validation
│   │   ├── tree_generation_prompts.py     # Enhanced LLM prompts
│   │   ├── semantic_chunker.py            # Smart chunking (planned)
│   │   ├── graph_nodes.py                 # LangGraph nodes
│   │   ├── langgraph_orchestrator.py      # Main workflow
│   │   └── ...
│   ├── models/
│   │   └── schemas.py                      # Enhanced models with routing
│   ├── streamlit_app/
│   │   ├── app.py                         # Main UI
│   │   └── components/
│   │       └── tree_visualizer.py         # Tile-based visualization
│   ├── a2a/
│   │   ├── agent.py                       # A2A agent
│   │   └── server.py                      # A2A server
│   └── database/
│       ├── models.py                      # SQLAlchemy models
│       └── operations.py                  # Database operations
├── docs/
│   ├── IMPROVEMENT_PLAN.md                # Roadmap
│   ├── IMPLEMENTATION_SUMMARY.md          # Current changes
│   ├── ARCHITECTURE.md                    # System design
│   └── QUICKSTART.md                      # Quick start
├── requirements.txt
├── main_a2a.py
└── README.md
```

## Enhanced Features

### Conditional Routing
All decision trees now include explicit routing logic:
- Each question specifies next node for every possible answer
- Support for complex conditions (AND/OR)
- Validation ensures all paths lead to outcomes
- No dead-end paths

### Path Validation
Automatic validation of tree structure:
- Checks for unreachable nodes
- Identifies incomplete routing
- Validates logic group consistency
- Reports path coverage statistics

### Semantic Chunking (Planned)
Intelligent document segmentation:
- Policy-aware boundary detection
- Preserves complete sections
- Maintains cross-references
- Adaptive chunk sizing

## Performance

### Benchmarks
- PDF parsing: 2-5 seconds
- Policy extraction: 10-30 seconds
- Tree generation: 5-15 seconds per tree
- Validation: <1 second per tree

### Optimization
- Parallel tree generation (up to 5 concurrent)
- Semantic chunking for better quality
- Confidence-based retry logic
- Redis caching for job state

## Troubleshooting

### Common Issues

**Trees show 0 questions**
- Solution: Check debug logs for parsing errors
- Verify tree structure with validator
- Ensure question extraction completed

**Incomplete routing warnings**
- Solution: Edit tree in Review tab
- Add missing answer paths
- Validate all outcomes are reachable

**High processing time**
- Solution: Use GPT-4o-mini for simple policies
- Enable semantic chunking
- Reduce chunk overlap

## Development

### Adding Custom Node Types
1. Update `NodeType` enum in schemas.py
2. Add handling in tree generator
3. Update visualization component
4. Add validation rules

### Extending Visualization
1. Modify `TileTreeVisualizer` class
2. Add custom render methods
3. Update styling in component
4. Test with real trees

## Testing

```bash
# Run all tests
pytest

# Test tree validation
pytest tests/test_tree_validator.py

# Test tree generation
pytest tests/test_decision_tree_generator.py

# Integration tests
pytest tests/test_integration.py
```

## Documentation

- **Quick Start**: `docs/QUICKSTART.md`
- **Architecture**: `docs/ARCHITECTURE.md`
- **Improvements**: `docs/IMPROVEMENT_PLAN.md`
- **Testing**: `docs/TESTING_GUIDE.md`

## Contributing

1. Fork repository
2. Create feature branch
3. Implement with tests
4. Update documentation
5. Submit pull request

## License

MIT License

## Support

- Issues: GitHub Issues
- Documentation: `docs/` directory
- Quick Start: `docs/QUICKSTART.md`
