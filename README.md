# 🚀 Hybrid RAG (v0.1.0): Advanced Retrieval & Generation Python Package

Build production-ready RAG solutions effortlessly with just a few lines of code! Hybrid RAG is designed to streamline your Retrieval-Augmented Generation (RAG) pipeline, making it easy to track, evaluate, and optimize your system for real-world applications, Explicit Asyncio support!!!!

### 🔥 Why Hybrid RAG?

✅ **Ingestion Pipeline** - End to End support to insert your vector data inside Milvus VectorDB. find the pipeline inside notebooks/ingestion_pipeline/Python_Feature_Pipeline.ipynb. In v0.1.1 support for kubeflow, pyspark pipelines to speed up Ingestion pipelines.
✅ **Experiment Tracking & Tracing with MLflow** – Log experiments, parameters, and traces for every LLM and retrieval step, ensuring efficient latency & cost tracking.
✅ **RAG Evaluation with Ragas** – Measure performance using faithfulness, answer relevance, and context precision, with future support for MLflow evaluation.
✅ **Cost Monitoring** – Keep track of API usage by setting LLM pricing inside API parameters to optimize expenses.
✅ **Hybrid Search Capability** – Semantic (dense) & keyword (sparse) retrieval, query expansion, Milvus-optimized retrieval, self-query retrieval, reranking, and auto-metadata filtering.
✅ **Nemo Guardrails (v0.1.1)** – Uses vector similarity for question classification, reducing middleware time, preventing prompt injection attacks, and enforcing policy restrictions.
✅ **Smart Summarization & Q&A Handling** – Supports direct QA over documents, metadata filtering, and map-reduce summarization for extracting insights across document chunks.
✅ **Follow-up Question Generation** – Auto-generate follow-up questions to improve engagement with users.
✅ **Custom PyFunc Hybrid-RAG MLflow Model** – Register, deploy, and serve the best model directly as an MLflow API for production-grade scenarios.
✅ **Optimized Modules with Async Code** – Fully asynchronous support for high-performance execution on Python 3.11+.
✅ **Speech-to-Text Model** – Supports local multilingual models, Hugging Face Inference API, and custom endpoints for speech-to-text conversion.
✅ **Enhanced Logger Support** – Detailed success/error logs stored in log/ with timestamped logs for full traceability.
✅ **Intelligent Modular Documentation** – Well-structured developer-friendly documentation with modular examples.
✅ **CI/CD Support** – Seamless model integration & deployment with GitHub Actions for Build-Test-Deploy pipelines.
✅ **Utility Functions for API/Streamlit apps** – Enables response storage on GitHub or AWS S3 for fine-tuning datasets and evaluation tracking.
✅ **Poetry, Makefile & Pre-commit Hooks** – Ensures best practices with pre-commit checks, packaging support, and agile development workflows.

Please Find detailed information about the strategy and usage of each module inside its respective README.md file. Each module has its own documentation to guide you through its functionality and implementation.

#### 🚀 Get Started in Minutes

```
    git clone <repo_url>
    cd hybrid_rag
    pip install poetry
    poetry build
    pip install dist/hybrid_rag-0.1.0-py3-none-any.whl
```

### 🔧 How to use Hybrid-RAG Package

Configure retrieval models, embeddings, and other settings based on your needs to build your advanced RAG pipeline in no time!
Now, forget about building Advance RAG from scratch —Hybrid RAG has got you covered! 🎯

> Code is fully tested and working properly, please let me know or add your queries in the discussion if you faced any issue.

#### Why a Python Package?

- Motivation behind building this package is Noticed reduction in latency of responses because of shifting from Simple code to more Async Modular code.
- It leverages easy configuration of any LLM model, parameters, retrieval embedding models to easily choose best hyperparameters for your data and application to setup.
- Reduced time and efforts to experiment with different parameters, to improve performance!

#### Does it Restricts with Less customization?

- No, Everything is customizable, you can configure the parameters easily putting everything inside .env.example file..

#### Am i able to Define my own flow for Hybrid-RAG steps?

- Yes, Absolutely every module is a separate Class, which leverage separate calling and make it more flexible and customizable...
- You can define different LLM models for different advance rag features like query expansion, self-query, rag chain, summary chain, followups etc.

This repo also provides support for a **Streamlit application and RestAPI support using FastAPI package**, please find their Setup Stratergy inside its respective README.md file inside chat_streamlit_app/* and chat_restapi/* leverages hybrid-rag package.

### Tech stack: 

- Python
- Langchain
- Nemo Guardrails
- RAGAS
- Milvus
- MongoDB
- LLM
- Semantic Search
- Mlflow - Tracing + Custom model logging
- Github Actions
- Transformers
- Hugging-Face
- Docker
- Streamlit
- FastAPI
- Asyncio
- AWS ECS - AWS EKS -> FastAPI and Streamlit deployemnt as docker container
- AWS Lambda - AWS API Gateway
- Package Building - Poetry, Makefile, Pre-Commit Hooks

### Latency Reduction Code Optimization Stratergies: 

- **Async packaging**: 
- **Cache for embedding models & LLM**: 
- **Generators**: 
- **Remove extra assignment of objects**: 
- **RunnableParallel**: 
- **Metadata Filtering**: 
- **Batch support**: 
- **Mlflow params cluster update**: 
- **delete the large outputs**: 

### Problems Faced and Improvise RAG: 

- **Hallucination in the responses** - multi-vector search by milvus does reranking Reranker() and context compression But by adding one more layer of reranking with a reranking model improvise the Context
- **In consistent responses** - Update Prompt and make it more robust with some explicit instructions to follow made responses more consistent
- **Knowledge chunk miss** - query expansion 
- **Complex queries** - 
- **Question type** - 
- **prompt-injestion** - 
- **No flow of conversation for personalised questions** - 
- **Latency** - 
- **speech to text capability** - 
- **Streamlit App response latency** - 

### Github Actions for CI/CD:

Github actions to Build-Test-Deploy Python Module and Deploy the Streamlit + FastAPI application as Docker Image inside AWS ECS.

Let's Understand, How you can Use this package to Built-Test-Deploy on your local environment.

below are the steps you can follow to run the Build-Test-deployment locally your package
 > Always run the package from your main-root folder i.e before hybrid_rag folder root directory

```
1. git clone https://github.com/kolhesamiksha/Hybrid-Search-RAG.git
1. cd Hybrid-Search-RAG
1. python -m venv myvenv
2. myvenv\Scripts\ativate
3. apt-get update && apt-get install -y --no-install-recommends \
    make build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    pip install poetry
4. poetry install --with lint,dev,typing,codespell
5. poetry build
6. make install-precommit
7. make run-precommit
8. make test
```

Now for Deployment of Streamlit and FastAPI Application developed a stratergy i.e. Deploy Both applications inside a single container and expose their ports independently one inside another using supervisord.

- Dockerfile - Created a flow to Expose a fastapi and then fruther added that inside streamlit application.

## Sample Output

I've builted a Chatbot over EY_IN Blogs data and test the end to end streamlit application builted using FastAPI which uses Hybrid-RAG Python package:)

![chatbot_img_5](https://github.com/user-attachments/assets/7f407d47-7ea6-492a-82bf-ff8d1b69b08c)
![chatbot_img_6](https://github.com/user-attachments/assets/ba6b6639-aef7-45f8-81b5-e7b9af6cf3e5)
![chatbot_img_7](https://github.com/user-attachments/assets/e23b4ede-9e60-4677-9589-d2702a15c837)
![chatbot_img_8](https://github.com/user-attachments/assets/3049bf88-6fce-4e1f-aea9-0f2897357fed)

📫 Developer contact:
Happy to Connect!! [Samiksha Kolhe](https://www.linkedin.com/in/samiksha-kolhe25701/)
