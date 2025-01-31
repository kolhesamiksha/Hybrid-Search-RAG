# 🚀 Hybrid RAG (v0.1.0): Advanced Retrieval & Generation Python Package

Build production-ready RAG solutions effortlessly with just a few lines of code! Hybrid RAG is designed to streamline your Retrieval-Augmented Generation (RAG) pipeline, making it easy to track, evaluate, and optimize your system for real-world applications, Explicit Asyncio support!!!!

![image](https://github.com/user-attachments/assets/2e48c31c-0a13-4cc9-8df8-e66e2fe10a50)

### 🔥 Why Hybrid RAG?

- ✅ **Ingestion Pipeline** - End to End support to insert your vector data inside Milvus VectorDB. find the pipeline inside notebooks/ingestion_pipeline/Python_Feature_Pipeline.ipynb. In v0.1.1 support for kubeflow, pyspark pipelines to speed up Ingestion pipelines.
- ✅ **Experiment Tracking & Tracing with MLflow** – Log experiments, parameters, and traces for every LLM and retrieval step, ensuring efficient latency & cost tracking.
- ✅ **RAG Evaluation with Ragas** – Measure performance using faithfulness, answer relevance, and context precision, with future support for MLflow evaluation.
- ✅ **Cost Monitoring** – Keep track of API usage by setting LLM pricing inside API parameters to optimize expenses.
- ✅ **Hybrid Search Capability** – Semantic (dense) & keyword (sparse) retrieval, query expansion, Milvus-optimized retrieval, self-query retrieval, reranking, and auto-metadata filtering.
- ✅ **Nemo Guardrails (v0.1.1)** – Uses vector similarity for question classification, reducing middleware time, preventing prompt injection attacks, and enforcing policy restrictions.
- ✅ **Smart Summarization & Q&A Handling** – Supports direct QA over documents, metadata filtering, and map-reduce summarization for extracting insights across document chunks.
- ✅ **Follow-up Question Generation** – Auto-generate follow-up questions to improve engagement with users.
- ✅ **Custom PyFunc Hybrid-RAG MLflow Model** – Register, deploy, and serve the best model directly as an MLflow API for production-grade scenarios.
- ✅ **Optimized Modules with Async Code** – Fully asynchronous support for high-performance execution on Python 3.11+.
- ✅ **Speech-to-Text Model** – Supports local multilingual models, Hugging Face Inference API, and custom endpoints for speech-to-text conversion.
- ✅ **Enhanced Logger Support** – Detailed success/error logs stored in log/ with timestamped logs for full traceability.
- ✅ **Intelligent Modular Documentation** – Well-structured developer-friendly documentation with modular examples.
- ✅ **CI/CD Support** – Seamless model integration & deployment with GitHub Actions for Build-Test-Deploy pipelines.
- ✅ **Utility Functions for API/Streamlit apps** – Enables response storage on GitHub or AWS S3 for fine-tuning datasets and evaluation tracking.
- ✅ **Poetry, Makefile & Pre-commit Hooks** – Ensures best practices with pre-commit checks, packaging support, and agile development workflows.

Please Find detailed information about the strategy and usage of each module inside its respective README.md file. Each module has its own documentation to guide you through its functionality and implementation.

### 🚀 Get Started in Minutes

```
    git clone https://github.com/kolhesamiksha/Hybrid-Search-RAG
    cd hybrid_rag
    pip install poetry
    poetry build
    pip install dist/hybrid_rag-0.1.0-py3-none-any.whl
```

### 🔧 How to use Hybrid-RAG Package

Configure retrieval models, embeddings, and other settings based on your needs to build your advanced RAG pipeline in no time!
Now, forget about building Advance RAG from scratch —Hybrid RAG has got you covered! 🎯

> Code is fully tested and working properly, please let me know or add your queries in the discussion if you faced any issue.

### 🛠️ Why a Python Package?

- The motivation behind creating this package was to reduce response latency by transitioning from simple code to an asynchronous, modular approach.
- It allows seamless configuration of LLM models, retrieval embeddings, and parameters, enabling you to fine-tune hyperparameters for your specific data and application.
- Saves time and effort in experimenting with different settings to enhance performance.

### 🔍 Is Customization Limited?

- Not at all! Everything is configurable. You can easily adjust the parameters using the .env.example file.

### 🏗️ Can I Define My Own Hybrid-RAG Workflow?

- Absolutely! Each module is designed as a separate class, making it highly flexible and customizable.
- You can assign different LLM models for various advanced RAG functionalities such as query expansion, self-query, RAG chains, summary chains, follow-ups, and more.
- Every Parameter and component is configurable.

### 📌 Additional Features
This repository also includes:

- **Streamlit Application** for interactive UI based Chat Interface which uses RestAPI to request and response through an API developed using Hybrid-RAG package..
- **FastAPI-based REST API** which makes use of Hybrid-RAG package 

**Knowledge Base**: Used EY-India Blogs as a Usecase to Build the Above system and used as a reference to build the package and streamlit application..

> NOTE: You can find setup instructions for both inside their respective README.md files in `chat_streamlit_app/` and `chat_restapi/`, all powered by the Hybrid-RAG package

### Tech stack: 

- Python
- Langchain
- RAG
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
- AWS Lambda - AWS API Gateway -> FastAPI deployment over AWS.
- Package Building - Poetry, Makefile, Pre-Commit Hooks

### Problems Faced and Improvise RAG: 

- **Hallucination in the responses** - multi-vector search by milvus does reranking Reranker() and context compression But by adding one more layer of reranking with a reranking model improvise the Context.
- **In consistent responses** - Update Prompt and make it more robust with some explicit instructions to follow made responses more consistent
- **Knowledge chunk miss** - query expansion enanles to create relevant questions based on the current question, which helps to improve the Context and chances of retrieving correct document and pass during Context preparation.
- **Complex queries** - At human level, as thinking differs person to person so as the complexity of queries.. Hence developed a Custom Query Expansion technique which caters such different types of questions.
- **Classify Questions** - From the market research and own understanding The data which i take reference from is of ey_in blogs data scraped from their allowed sitemaps. in the current version developed 2 systems/approaches to cater 2 types of queries those are Blogs summarization questions and Normal blogs questions..
- **No flow of conversation for personalised questions** - During trying of the chatbot, i noticed people also asked normal personal level questions or questions which are not related to the KnowldgeBase, For normal questions there should be flow define and not requires RAG to be get triggered everytime. 
- **Prompt-Injestion** - For Production Usecase or public chatbots which uses LLM's, people faced issues of jailbreaking and prompt-injestion.. Hence to add a Security layer to block hitting LLM and Hybrid-RAG system, develope a stratergy using Nemo Guardrails (avaialble in v0.1.1).
- **Latency** - This is the Major issue i faced during the developement of complex systems, everytime when you hit the question the hybrid-rag with all local embedding downloading and llm api calls happen in the backend, hence Implemented caching for Local sparse-dense and llm calls, async codebase helps reducing latency and blocking for awaited independent tasks.
- **Speech to text capability** - Speech to text model for real-time use of chatbot over voice input questions..
- **Performance Tracking** - mlflow performance tracking can be developed
- **Streamlit App response latency** - I compared the streamlit application, with Using Hybrid-RAG package, Mlflow logged model as URI, Using FastAPI for request response.. with Using FastAPI seems less latency in response and less time required. Please see the results below

### Latency Reduction Code Optimization Stratergies: 

- **Async packaging**: As Code is more API bound, Waiting time for a particular module was blocking other independent functions, made a lot of waiting and in-efficient CPU utilizatio - 30% time reduction.
- **Cache for embedding models & LLM**: Used Local Embedding models mostly used for hybrid-searching, these models used to get loaded for each query searching, causes increased latency and memory consumption.. with lru_cache mechanism significantly reduce latency and reduce the processing time.
- **Generators**: Generator helps to reduce memory consumption, using memory_profiler adding generators mainly as a list comprehensions, storing larger data i.e. data retrieved and combined from all expanded queries before sending to reranker + summarisation casde study which stores all documents chunks retrieved for summarization.. Reduced memory consmuption upto 20%. 
- **Remove extra assignment of objects**: remove extra assignement of variables, reduces the cyclic variables.. as python doesnot remove such assignments during garbage collection, this helps code looks more structures, less complex and more comprehensive.
- **RunnableParallel**: Implementing runnable parallel, Noticed 20% reduction in chain-time instead of passing everything sequentially.
- **Metadata Filtering**: Metadata filtering reduces efforts for searching of data from entire database to only a subset of data, which improves accuracy, correct chunks in the context and indirectly reduces hallucinations, helps mostly in summarization case study
- **Batch support**: Implemented batch processing
- **Bathc Mlflow params**: previously logging every parameter separately, consumes time and compute to first serialise those params as per mlflow utf-8 support & contact backend mlflow database for storage.. by batching, at a single mlflow call and validation reduced cpu preemption and reduce the memory consumption.
- **Delete the large outputs**: This reduced Significant memory consumption as per profiler report.. notices 40% reduction in memory consumpion. 

`constantly working on implementing more practices to reduce memory consumption, reduce latency and improvise CPU utilization`

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
