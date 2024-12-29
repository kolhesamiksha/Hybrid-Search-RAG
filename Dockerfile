# Use Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Poetry and Supervisor
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl make build-essential supervisor && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Copy project files into the container
COPY . .

# Build the wheel file using the Makefile
RUN make build

# Install the built wheel file
RUN pip install dist/*.whl

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Copy Supervisor configuration
RUN mkdir -p /etc/supervisor/conf.d
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Run Supervisor to manage multiple processes (FastAPI + Streamlit)
CMD ["supervisord", "-n"]
