# Policy Analysis Agent

The Policy Analysis Agent extracts medical policy criteria from policy documents and generates corresponding questionnaires for prior authorization processes.

## Features

- **Policy Document Analysis**: Extracts key criteria and requirements from medical policy documents
- **Questionnaire Generation**: Creates structured questionnaires based on extracted criteria
- **Prior Authorization Support**: Specialized for medical prior authorization workflows
- **A2A Protocol Integration**: Full integration with Agent-to-Agent communication protocol

## Configuration

The agent uses environment variables prefixed with `PA_` for configuration:

### Required Variables
- `PA_LLM_API_KEY`: API key for LLM service
- `PA_LLM_ENDPOINT`: Endpoint for LLM service (default: https://smartops-llmops.eastus.cloudapp.azure.com/litellm)
- `PA_LLM_MODEL`: Model to use for LLM service (default: azure/sc-rnd-gpt-4o-mini-01)

### Optional Variables
- `PA_AGENT_NAME`: Name of the agent (default: Policy Analysis Agent)
- `PA_AGENT_HOST`: Agent host (default: localhost)
- `PA_AGENT_PORT`: Agent port (default: 10001)
- `PA_AGENT_VERSION`: Agent version (default: 1.0.0)
- `PA_AGENT_DESCRIPTION`: Agent description

## Usage

### Local Development
```bash
cd policy_analysis_agent
python main.py --host 0.0.0.0 --port 10001
```

### Docker Deployment
```bash
docker build -t policy-analysis-agent .
docker run -p 10001:10001 --env-file .env policy-analysis-agent
```

### Local Development with pip
```bash
pip install -e .
cd policy_analysis_agent
python main.py --help  # See all available options
python main.py --log-level DEBUG  # Start with debug logging
```

## Agent Capabilities

### Skills
- **analyze_policy**: Analyzes medical policy documents to extract criteria and generate questionnaires

### Input Format
The agent accepts policy documents in text format and processes them to extract:
- Medical criteria and requirements
- Coverage conditions
- Prior authorization requirements
- Documentation requirements

### Output Format
Returns structured data including:
- Extracted criteria
- Generated questionnaire questions
- Policy analysis results
- Status information

## Docker Deployment

### Quick Start with Docker Compose
```bash
# Copy and configure environment variables
cp .env.example .env
# Edit .env with your configuration

# Build and start the service
docker-compose up --build -d

# View logs
docker-compose logs -f policy-analysis-agent

# Stop the service
docker-compose down
```

### Manual Docker Build
```bash
# Build the image
docker build -t policy-analysis-agent .

# Run the container
docker run -d \
  --name policy-analysis-agent \
  -p 10001:10001 \
  --env-file .env \
  policy-analysis-agent

# View logs
docker logs -f policy-analysis-agent
```

### Health Check
The service includes a health check endpoint:
```bash
curl http://localhost:10001/health
```

## Development

### File Structure
```
policy-analysis-agent/
├── policy_analysis_agent/
│   ├── agent.py              # Main agent logic
│   ├── agent_executor.py     # A2A protocol handler
│   ├── main.py              # Click CLI application
│   ├── docker_entrypoint.py # Container entrypoint
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

## Environment Variables

Required environment variables (set in `.env` file):

```bash
# Required - LLM Configuration
PA_LLM_API_KEY=your_api_key_here
PA_LLM_ENDPOINT=https://smartops-llmops.eastus.cloudapp.azure.com/litellm
PA_LLM_MODEL=azure/sc-rnd-gpt-4o-mini-01

# Optional - Agent Configuration
PA_AGENT_NAME=Policy Analysis Agent
PA_AGENT_HOST=localhost
PA_AGENT_PORT=10001
PA_AGENT_VERSION=1.0.0
PA_AGENT_DESCRIPTION=Extracts medical policy criteria from policy documents and generates questionnaires

# Optional - Container Configuration  
LOG_LEVEL=INFO
```

## License

This project is part of the GenHealth.ai system.