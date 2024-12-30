###################################### Stage 1: BUILD ######################################

# Use Python base image
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /build

# Install system dependencies for Poetry
RUN apt-get update && apt-get install -y --no-install-recommends \
    make build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    pip install poetry

# Copy project files into the container
# hybrid_rag: SIZE: 0.19MB
COPY hybrid_rag hybrid_rag
COPY tests tests
COPY .pre-commit-config.yaml .pre-commit-config.yaml
COPY Makefile Makefile
COPY poetry.toml poetry.toml
COPY pyproject.toml pyproject.toml

# Optional: If No workflows/cicd setup for build test and deploy.. then use make install, make install-precommit, make run-precommit, make test, make clean etc
# Build the wheel file using the Makefile
RUN make build

###################################### Stage 2: RUNTIME ###########################################
FROM python:3.11-slim

# Set working directory
WORKDIR /Hybrid-Search-RAG

# Install system dependencies for Poetry and Supervisor
RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/dist/*.whl .

# Install the wheel file && remove the wheel after installation
RUN pip install *.whl && rm -rf *.whl

##################
# chat_restapi -> SIZE: 0MB
# chat_streamlit_app -> SIZE: 0.23MB
##################

# Copy project files into the container
COPY chat_restapi chat_restapi
COPY chat_streamlit_app chat_streamlit_app
COPY .env.example chat_restapi/.env.example
COPY .env.example chat_streamlit_app/.env.example

#Supervisord.conf COPY
RUN mkdir -p /etc/supervisor/conf.d
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Run Supervisor to manage multiple processes (FastAPI + Streamlit)
CMD ["supervisord", "-n"]
