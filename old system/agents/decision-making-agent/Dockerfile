# Base image
ARG Base_Image=smartopsdev.azurecr.io/so-infra/smartops_python_3.11_ubu22.04:latest
FROM ${Base_Image}

# Install required system packages including curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install venv package required for pipx
RUN apt-get update && apt-get install -y python3.11-venv && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3.11 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --no-cache-dir --upgrade pip

# Install dependencies from pyproject.toml
ADD pyproject.toml /app/
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --no-cache-dir -e .

# Copy app code
ADD . /app/

# Set working directory for your service
WORKDIR /app/decision_making_agent

# Expose service port
EXPOSE 10003

# Use uvicorn directly since it's working well with proper logging
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10003", "--log-level", "info"]