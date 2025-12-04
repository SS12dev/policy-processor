# Decision Making Agent

The Decision Making Agent processes prior authorization decisions based on policy criteria and patient application data.

## Features

- **Authorization Decision Making**: Makes informed prior authorization decisions
- **Policy Compliance**: Evaluates applications against policy criteria
- **Evidence Assessment**: Analyzes medical evidence and documentation
- **Decision Rationale**: Provides clear reasoning for approval/denial decisions
- **A2A Protocol Integration**: Full integration with Agent-to-Agent communication protocol

## Configuration

The agent uses environment variables prefixed with `DM_` for configuration:

### Required Variables
- `DM_LLM_API_KEY`: API key for LLM service
- `DM_LLM_ENDPOINT`: Endpoint for LLM service (default: https://smartops-llmops.eastus.cloudapp.azure.com/litellm)
- `DM_LLM_MODEL`: Model to use for LLM service (default: azure/sc-rnd-gpt-4o-mini-01)

### Optional Variables
- `DM_AGENT_NAME`: Name of the agent (default: Decision Making Agent)
- `DM_AGENT_HOST`: Agent host (default: localhost)
- `DM_AGENT_PORT`: Agent port (default: 10003)
- `DM_AGENT_VERSION`: Agent version (default: 1.0.0)
- `DM_AGENT_DESCRIPTION`: Agent description

## Usage

### Local Development
```bash
cd decision_making_agent
python main.py --host 0.0.0.0 --port 10003
```

### Docker Deployment
```bash
docker build -t decision-making-agent .
docker run -p 10003:10003 --env-file .env decision-making-agent
```

### Local Development with pip
```bash
pip install -e .
cd decision_making_agent
python main.py --help  # See all available options
python main.py --log-level DEBUG  # Start with debug logging
uvicorn main:app --host 0.0.0.0 --port 10003
```

## Agent Capabilities

### Skills
- **make_authorization_decision**: Makes prior authorization decisions based on policy criteria and patient application data

### Input Format
The agent accepts:
- Policy criteria and requirements
- Patient application data
- Medical evidence and documentation
- Prior authorization request details

### Output Format
Returns structured decisions including:
- Authorization decision (approve/deny/pending)
- Decision rationale and reasoning
- Required additional information (if applicable)
- Compliance assessment results

## Decision Logic

The agent evaluates applications based on:
1. **Policy Compliance**: Checks if patient meets policy criteria
2. **Medical Necessity**: Assesses medical justification
3. **Documentation Quality**: Reviews supporting evidence
4. **Risk Assessment**: Evaluates potential risks and benefits

## Environment Variables

Copy `.env.example` to `.env` and update with your configuration:

```bash
cp .env.example .env
```

## License

This project is part of the GenHealth.ai system.