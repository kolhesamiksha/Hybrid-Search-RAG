# Use the AWS-provided base image for Python
FROM public.ecr.aws/lambda/python:3.11

# Set the working directory
WORKDIR ${LAMBDA_TASK_ROOT}

RUN mkdir chat_restapi

# Copy your function code and other required files to the container
COPY router chat_restapi/router/
COPY main.py chat_restapi/main.py
COPY requirements.txt chat_restapi/requirements.txt

COPY hybrid_rag-0.1.0-py3-none-any.whl chat_restapi/hybrid_rag-0.1.0-py3-none-any.whl
# Install dependencies
RUN pip install --no-cache-dir -r chat_restapi/requirements.txt \
    && pip install --no-cache-dir chat_restapi/hybrid_rag-0.1.0-py3-none-any.whl

# Specify the Lambda handler
CMD ["chat_restapi.main.handler"]
