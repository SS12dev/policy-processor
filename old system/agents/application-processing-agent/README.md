# Application Processing Agent

The Application Processing Agent extracts and validates medical application data from patient documents for prior authorization processes.

## Features

- **Application Data Extraction**: Extracts patient information and medical data from application documents
- **Data Validation**: Validates extracted data against required fields and formats
- **Patient Summary Generation**: Creates structured patient summaries from application data
- **A2A Protocol Integration**: Full integration with Agent-to-Agent communication protocol

## Configuration

The agent uses environment variables prefixed with `AP_` for configuration:

### Required Variables
- `AP_LLM_API_KEY`: API key for LLM service
- `AP_LLM_ENDPOINT`: Endpoint for LLM service (default: https://smartops-llmops.eastus.cloudapp.azure.com/litellm)
- `AP_LLM_MODEL`: Model to use for LLM service (default: azure/sc-rnd-gpt-4o-mini-01)

### Optional Variables
- `AP_AGENT_NAME`: Name of the agent (default: Application Processing Agent)
- `AP_AGENT_HOST`: Agent host (default: localhost)
- `AP_AGENT_PORT`: Agent port (default: 10002)
- `AP_AGENT_VERSION`: Agent version (default: 1.0.0)
- `AP_AGENT_DESCRIPTION`: Agent description

## Docker Deployment

### Quick Start with Docker Compose
```bash
# Copy and configure environment variables
cp .env.example .env
# Edit .env with your configuration

# Build and start the service
docker-compose up --build -d

# View logs
docker-compose logs -f application-processing-agent

# Stop the service
docker-compose down
```

### Manual Docker Build
```bash
# Build the image
docker build -t application-processing-agent .

# Run the container
docker run -d \
  --name application-processing-agent \
  -p 10002:10002 \
  --env-file .env \
  application-processing-agent

# View logs
docker logs -f application-processing-agent
```

### Health Check
The service includes a health check endpoint:
```bash
curl http://localhost:10002/health
```

## Local Development

### With uvicorn
```bash
cd application_processing_agent
uvicorn main:app --host 0.0.0.0 --port 10002
```

### With enhanced CLI
```bash
cd application_processing_agent
python main.py --host 0.0.0.0 --port 10002 --log-level INFO
```

### With pip install
```bash
pip install -e .
cd application_processing_agent
python main.py --help  # See all available options
```

## Environment Variables

Required environment variables (set in `.env` file):

```bash
# Required - LLM Configuration
AP_LLM_API_KEY=your_api_key_here
AP_LLM_ENDPOINT=https://smartops-llmops.eastus.cloudapp.azure.com/litellm
AP_LLM_MODEL=azure/sc-rnd-gpt-4o-mini-01

# Optional - Agent Configuration
AP_AGENT_NAME=Application Processing Agent
AP_AGENT_HOST=localhost
AP_AGENT_PORT=10002
AP_AGENT_VERSION=1.0.0
AP_AGENT_DESCRIPTION=Extracts and validates medical application data from patient documents

# Optional - Container Configuration  
LOG_LEVEL=INFO
```

## Agent Capabilities

### Skills
- **extract_application_data**: Extracts and validates patient data from medical application documents

### Input Format
The agent accepts application documents and processes them to extract:
- Patient demographic information
- Medical history and conditions
- Treatment requests
- Insurance information
- Supporting documentation

### Output Format
Returns structured data including:
- Extracted patient data
- Validation results
- Patient summary
- Data quality scores

## Development

### File Structure
```
application-processing-agent/
├── application_processing_agent/
│   ├── agent.py              # Main agent logic
│   ├── agent_executor.py     # A2A protocol handler
│   ├── main.py              # Click CLI application
│   ├── config/
│   │   └── settings.json    # Agent configuration schema
│   └── utils/
│       ├── llm.py          # LLM integration
│       └── settings.py     # Configuration management
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── pyproject.toml
├── .env
└── README.md
```

## License

This project is part of the GenHealth.ai system.