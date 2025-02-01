# 🚀 Hybrid RAG (v0.1.0): Advanced Retrieval & Generation Python Package

Build production-ready RAG solutions effortlessly with just a few lines of code! Hybrid RAG is designed to streamline your Retrieval-Augmented Generation (RAG) pipeline, making it easy to track, evaluate, and optimize your system for real-world applications, Explicit Asyncio support!!!!

![image](https://github.com/user-attachments/assets/2e48c31c-0a13-4cc9-8df8-e66e2fe10a50)


<span style="background-color: yellow;"> If you wanted in-depth understanding of Advance RAG, Code Packaging and best practices to built efficient and scalable RAG, I'll be writing the blogs on each and every concept used in this project and how one can also start building such packaged systems.. Please Subscribe or follow my blogs community, to read them.. link provided at the End!!</span>.


### 🔥 Why Hybrid RAG?
---

- ✅ **Ingestion Pipeline** - End to End support to insert your vector data inside Milvus VectorDB. find the pipeline inside notebooks/ingestion_pipeline/Python_Feature_Pipeline.ipynb. In v0.1.1 support for kubeflow, pyspark pipelines to speed up Ingestion pipelines.
- ✅ **Experiment Tracking & Tracing with MLflow** – Log experiments, parameters, and traces for every LLM and retrieval step, ensuring efficient latency & cost tracking.
- ✅ **RAG Evaluation with Ragas** – Measure performance using faithfulness, answer relevance, and context precision, with future support for MLflow evaluation.
- ✅ **Cost Monitoring** – Keep track of API usage by setting LLM pricing inside API parameters to optimize expenses.
- ✅ **Hybrid Search Capability** – Semantic (dense) & keyword (sparse) retrieval, query expansion, Milvus-optimized retrieval, self-query retrieval, reranking, and auto-metadata filtering.
- ✅ **Nemo Guardrails (v0.1.1)** – Uses vector similarity for question classification, reducing middleware time, preventing prompt injection attacks, and enforcing policy restrictions. available in v0.1.1
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

---

### 🚀 Get Started in Minutes

```
    git clone https://github.com/kolhesamiksha/Hybrid-Search-RAG
    cd Hybrid-Search-RAG
    pip install poetry
    poetry build
    pip install dist/hybrid_rag-0.1.0-py3-none-any.whl
```
**Note: I'll provide the blog links for each of the features discussed in this repo in future, Stay tuned and fork the repository for future updates...**

### 🔧 How to use Hybrid-RAG Package

Configure retrieval models, embeddings, and other settings based on your needs to build your advanced RAG pipeline in no time!
Now, forget about building Advance RAG from scratch —Hybrid RAG has got you covered! 🎯

> Code is fully tested and working properly, please let me know or add your queries in the discussion if you faced any issue.

**Please find the sample examples in examples/* to use this package for your usecase**

### 🛠️ Why a Python Package?

- The motivation behind creating this package was to reduce response latency by transitioning from simple code to an asynchronous, modular approach.
- It allows seamless configuration of LLM models, retrieval embeddings, and parameters, enabling you to fine-tune hyperparameters for your specific data and application.
- Saves time and effort in experimenting with different settings to enhance performance.


### 🔍 Is Customization Limited?

- Not at all! Everything is configurable. You can easily adjust the parameters using the .env.example file.
- Proper comments are provided inside .env.example file to find and get every configurations..
- Start Build and deploy your usecase without caring about the internal techstack with just few lines of code!!!


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

---

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

### Problems Faced and their Solutions: 
---

- **Hallucination in the responses** - multi-vector search by milvus does reranking Reranker() and context compression But by adding one more layer of reranking with a reranking model improvise the Context.

- **In consistent responses** - Update Prompt and make it more robust with some explicit instructions to follow made responses more consistent.

- **Knowledge chunk miss** - query expansion enanles to create relevant questions based on the current question, which helps to improve the Context and chances of retrieving correct document and pass during Context preparation.

- **Complex queries** - At human level, as thinking differs person to person so as the complexity of queries.. Hence developed a Custom Query Expansion technique which caters such different types of questions.

- **Classify Questions** - From the market research and own understanding The data which i take reference from is of ey_in blogs data scraped from their allowed sitemaps. in the current version developed 2 systems/approaches to cater 2 types of queries those are Blogs summarization questions and Normal blogs questions..

- **No flow of conversation for personalised questions** - During trying of the chatbot, i noticed people also asked normal personal level questions or questions which are not related to the KnowldgeBase, For normal questions there should be flow define and not requires RAG to be get triggered everytime.

- **Prompt-Injestion** - For Production Usecase or public chatbots which uses LLM's, people faced issues of jailbreaking and prompt-injestion.. Hence to add a Security layer to block hitting LLM and Hybrid-RAG system, develope a stratergy using Nemo Guardrails (avaialble in v0.1.1).

- **Latency** - This is the Major issue i faced during the developement of complex systems, everytime when you hit the question the hybrid-rag with all local embedding downloading and llm api calls happen in the backend, hence Implemented caching for Local sparse-dense and llm calls, async codebase helps reducing latency and blocking for awaited independent tasks.

- **Speech to text capability** - Speech to text model for real-time use of chatbot over voice input questions..

- **Performance Tracking** - performance tracking by configuring parameters was harder to track and trace the effect of approaches used to reduce chain times of different components of hybrid-rag.

- **Streamlit App response latency** - I compared the streamlit application, with Using Hybrid-RAG package, Mlflow logged model as URI, Using FastAPI for request response.. with Using FastAPI seems less latency in response and less time required. Please see the results below

![image](https://github.com/user-attachments/assets/d81298bb-a261-4ef6-8c29-8a2b15f0726d)

### Mlflow Traceability

- You can see full trace of your RAG steps each and every step where your llm model is getting used...

![image](https://github.com/user-attachments/assets/8e2636a1-5360-40d7-bcfd-ce14e6b42eb6)

### Mlflow logged Metrices and Params

- You can get all relevant parameters to track your experiements, accuracy and register the Best Model...

> Note: To register the best model after experimentation, please run `python examples/custom_mlflow_rag_model_logging.py`, this will logged the RAG model and you can serve that as API for your production usecase.

![image](https://github.com/user-attachments/assets/5306c077-d5de-4c19-89b8-cd1be4483dc2)


### Latency Reduction Code Optimization Stratergies: 
---

- **Asynchronous Codebase**: Since the code is heavily API-driven, waiting times for specific modules were blocking other independent functions, leading to inefficiencies in CPU utilization. Implementing asynchronous execution resulted in a 30% reduction in processing time.

- **Caching for Embedding Models & LLM**: Previously, local embedding models—primarily used for hybrid searching—were reloaded for each query, increasing both latency and memory consumption. By implementing an LRU cache mechanism, we significantly reduced query latency and optimized processing time.

- **Utilizing Generators**: Generators were introduced to optimize memory usage, particularly when handling large datasets retrieved from multiple expanded queries before reranking and summarization. Using memory_profiler, we strategically replaced list comprehensions with generators, resulting in a 20% reduction in memory consumption.

- **Eliminating Redundant Object Assignments**: By removing unnecessary variable assignments, we minimized cyclic references that Python's garbage collector does not automatically clean up. This enhanced the structural clarity of the code, making it more concise, efficient, and maintainable.

- **Implementing RunnableParallel**: Transitioning from sequential execution to RunnableParallel led to a 20% reduction in overall chain execution time, significantly improving performance.

- **Metadata Filtering for Targeted Retrieval**: Introducing metadata filtering improved data search efficiency by restricting queries to relevant subsets rather than searching the entire database. This enhanced contextual accuracy, reduced hallucinations, and improved summarization quality.

- **Batching MLflow Parameters**: Previously, logging each MLflow parameter separately incurred serialization overhead and frequent backend interactions, increasing compute load. By batching parameters into a single MLflow call, we reduced CPU preemption and optimized memory usage.

- **Deleting Large Outputs**: Proactively clearing large, unused outputs significantly reduced memory overhead. As observed in profiling reports, this approach led to a 40% reduction in memory consumption.

`constantly working on implementing more practices to reduce memory consumption, reduce latency and improvise CPU utilization`

### Github Actions for CI/CD:
---

GitHub Actions for Building, Testing, and Deploying a Python Module, and Deploying Streamlit + FastAPI as a Docker Image on AWS ECS  

#### Local Build, Test, and Deployment Guide  

This guide provides step-by-step instructions to build, test, and deploy your package locally.  

### **Prerequisites**  
- Ensure you are running all commands from the **root directory**, i.e., the parent directory of `hybrid_rag`.  

### **Steps to Build, Test, and Deploy Locally**  

```sh
1. git clone https://github.com/kolhesamiksha/Hybrid-Search-RAG.git
2. cd Hybrid-Search-RAG
3. python -m venv myvenv
4. myvenv\Scripts\activate  # For Windows users
   source myvenv/bin/activate  # For macOS/Linux users
5. apt-get update && apt-get install -y --no-install-recommends \
   make build-essential && \
   apt-get clean && rm -rf /var/lib/apt/lists/* && \
   pip install poetry
6. poetry install --with lint, dev, typing, codespell
7. poetry build
8. make install-precommit
9. make run-precommit
10. make test
```

#### Unique Deployment Strategy for Streamlit and FastAPI as Docker Contaienr over AWS

To optimize deployment, both Streamlit and FastAPI applications are hosted within a single container, with each service exposing its ports independently. Supervisord is used to manage both processes efficiently.

**Dockerfile Design**

- The FastAPI application is exposed first, serving as the backend.
- The Streamlit application is integrated within the same container, ensuring seamless interaction with FastAPI.
- The Docker image follows best practices to be lightweight, ensuring minimal memory consumption and faster build times.

This approach ensures an efficient, scalable, and production-ready deployment. 🚀

I'll provide the details and best practices to built the image in future, Please Stay tuned and fork the repository for future updates...

## Sample Streamlit Application Output

I've builted a Chatbot over **EY-India Blogs data** Scraped from their Website(Scraping was allowed by their robots.txt) and test the end to end streamlit application builted using streamlit as frontend and Request-response using a RestAPI builted using FastAPI which uses Hybrid-RAG as Python package:)

![chatbot_img_5](https://github.com/user-attachments/assets/7f407d47-7ea6-492a-82bf-ff8d1b69b08c)
![chatbot_img_6](https://github.com/user-attachments/assets/ba6b6639-aef7-45f8-81b5-e7b9af6cf3e5)
![chatbot_img_7](https://github.com/user-attachments/assets/e23b4ede-9e60-4677-9589-d2702a15c837)
![chatbot_img_8](https://github.com/user-attachments/assets/3049bf88-6fce-4e1f-aea9-0f2897357fed)

## 📫 Developer Contact:

**Samiksha Kolhe:)**

I’d love to connect and collaborate! Feel free to reach out. 🚀

🔗 [LinkedIn](https://www.linkedin.com/in/samiksha-kolhe25701/)
⭐ [GitHub](https://github.com/kolhesamiksha)
📊 [Kaggle](https://www.kaggle.com/samikshakolhe)

If you found this project helpful, don’t forget to star ⭐ the repository and fork it for future updates! 🚀

Stay ahead in AI and keep up with the latest tech trends by subscribing to my blog community:
📝 [Teckbakers](https://teckbakers.hashnode.dev/) — Your hub for cutting-edge AI insights and innovations!
