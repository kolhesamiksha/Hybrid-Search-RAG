# Hybrid-Search-RAG-Chatbot

This Chatbot Application is builted using Advance RAG Techniques, It helps to answer Queries regarding Technology and Latest trends in the Market. Vast amount of Latest trends, knowledge/case studies from Domain Experts and blogs was available, collected from ey website.   

![advance-rag-workflow](https://github.com/user-attachments/assets/f66d9c12-5356-4b48-8a2e-1a4551181f57)

### Deployed FASTAPI: 

https://comparable-clarie-adsds-226b08fd.koyeb.app/

### Deployed Streamlit app

https://hybrid-search-rag-chatbot-sam.streamlit.app/

## Approach: RAG Workflow Stratergy

The approaches Followed for building RAG pipeline is discussed below

#### 1. WebCrawling: 

The Website was a rich source of documents in everydomain like AI, Supply chain, Digital, Cybersecurity and Various Informational Blogs and Use-cases Discussed. Most of the websites have a common web interface and HTML Codebase which was suitable to built an automation using For Webscraping. 
1. Used Sitemap of Topics under ey_in domain.
1. Analyse the Structure of the Websites.
2. Extracted metadata required further for metadata filtering or Self-Query Retriever to post-retrieval for better performance i.e. Author name, related topics, pdf_links. Pdf links added because most of the blogs contains PDF for detailed information. in future content can be extracted and used for training of the chatbot for more information.
3. Built an Automation to scrape and extract 2k websites using unstructuredURLLoader by langchain.

#### 2. Semantic Chunking Techniques:

After WeCrawling and Scraping data from the BaseURL. Applied various Techniques like RecursiveCharacterTextSplitter, Semantic Chunking, Statistical Chunking, Rolling Window Spltting using Semantic-routing.

1. **RecursiveCharacterTextSplitter**: It splits the data by intelligently analysing the structure of the data based on splitting criterias. But Limitation is Not Calculate the semantic similarity between the context while splitting. Hence, Not better if we are unknown to the fact of Structure of Data.

2. **Semantic Chunking**: Semantic Chunking splits the data in between the sentences. Based in the similarity between the sentences it combines the sentences and split the data where similarity drops to an extend. It works on embedding Similarity between the Sentences. 

- 2.1 **Statistical Chunking**: Better for English Text only. Instead of chunking text with a fixed chunk size, the semantic splitter adaptively picks the breakpoint in-between sentences using embedding similarity. This ensures that a "chunk" contains sentences that are semantically related to each other. For Semantic chunking used **jinaai/jina-embeddings-v2-base-en** (8K context length) by langchain FastEmbedding Module.

- 2.2 **Rolling Window Spltting(Used in RAG)**: It uses a rolling window to consider splitting and applies semantic similarity while considering the sentence to combine and split. This Technique is more generic for any type of embeddding model, MAX_SPLI, MIN_SPLIT parameters makes it more customisable. Providing Chunks compatible to semantic chunking technique.

#### 3. Metadata Chunking Method: 

Metadata Filtering is a way to limit the searches and increase chances of Information exact retrieval of chunks. For Metadata added Primary Source_links, author_names, related_topics, pdf_links.

#### 4: Embedding Models:
Tried Different types of embedding models by considering system Size and Best performance for english text on MTEB. Used Fastembedding() from langchain to get the embeddings best models available for local instead of API. Local Embedding Hosting saves the credits and Manage the latency during retrieval pipeline.

Used Embedding model(Hybrid-search): SPARSE_EMBEDDING_MODEL: **Qdrant/bm42-all-minilm-l6-v2-attentions**, DENSE_EMBEDDING_MODEL: **jinaai/jina-embeddings-v2-base-en**.

#### 5. Vector-Store: Zillinz hosted Milvus Store:

Used Milvus to store the emebeddings, pymilvus module is more customisable for hybrid search and more scalable with Task Specific emebeddng indexes available for dense and sparse embeddings.

#### 6. Query-Expansion Techniques: Self-Query retrieval for metadata & MultiQuery:

Query Compression techniques are like Query breakdown, Query exapansion(Multiple Queries). Created a Customised MultiQuery Retrieval Class find on Chatbot-streamlit/src/utils/custom_utils.py. Defined my own prompt for Query formulations and breakdown. 

#### 7. Metadata-Filtering techniques: 

For Metadata Filtering **Used Self-Query Retrieval** which used LLM model to get the **filters and strctured query** relevant to Query by the User.

#### 8. Retrieval: Hybrid-search:

Used Hybrid Search By milvus, Stored Sparse and Dense vectors indexes inside the milvus collection. During retrieval used ANNSSearch to retrieve the Chunks. 
Applied Hybrid Search on Multiple queries generated by Query-expansion techniques + Metadata-filtering by Self-Query Over Sparse & Dense embedding search Limit 3 Each. 
Total for 5 queries using Sparse search generated: 15 chunks & using Dense search generated: 15 chunks Subtotal 30 Chunks Retrieved. 

#### 9. Reranking Techniques:

Reranking technique is Important and Useful When applied Selfquery & Multi query generation technique to Re-rank the Chunks and Retrieve most Ranked with High similairty To consider and send as Context to LLM.

Re-ranking Models: Used Flashrank defautt Local Reranking model:**ms-macro-tinybert-l-2-v2**. SPLADE/ColBert are most used reranking models and perform best.

#### 10. LLM Model:

LLM Model: Used gpt-4o model using AI/ML API. Best performing LLM model available. Used for Smooth Deployment to make it public

## Results

**Features**:
- Post meta filtering
- Hybrid-Search
- Chat-history as 2
- Sourcelinks in the Response
- Latency(time require to process the Request) , Cost-per-request in the Response.

## Code Structure

```
    Chatbot-streamlit/
        ├── css/
        │ └── style.css
        ├── src/
        │ ├── logs/
        │ ├── utils/
        │ │ ├── init.py
        │ │ ├── custom_utils.py
        │ │ ├── get_insert_mongo_data.py
        │ │ ├── logutils.py
        │ │ └── mongo_init.py
        │ ├── api.py
        │ ├── main.py
        │ ├── schema.py
        │ └── test.py
        ├── ui/
        │ ├── sidebar.md
        │ └── styles.md
        ├── app.py
        ├── requirements.txt
        └── vercel.json
    images/
        ├── advance-rag-workflow.PNG
    Milvus_features_pipeline/
        ├── Milvus_Feature_Pipeline_Final.ipynb
    Research_Notebooks/
        ├── Milvus_Feature_Pipeline_sample.ipynb
    Retriever_Pipeline/
        ├── Retriever_Pipeline.ipynb
```

## TechStack

- Python
- Web Scraping
- MongoDB (Store API Secrets)
- Generative AI
- RAG Chatbot
- Vector Search (Hybrid-Filtering)
- Langchain
- Semantic Router

## How to Run the Application

> Note: Code is builted and tested on python==^3.11.5

This Application is Not yet deployed.. In-progress.

Follow Below Instruction for smooth and Errorless Application Run
**Setup the Virtual-env**

```
    conda create --name ragapp python=3.11.5
    conda activate ragapp

    cd Chatbot-streamlit/
    pip install -r requirements.txt
```

### Setup the API Backend

To Run the Application Locally First Run the FastAPI backend. Follow Below Instructions to Run the FastAPI:

**Note**: To add your credentials inside the Mongodb Atlas cloud to connect the application Refere: Chatbot-streamlit/src/utils/mongo_init.py

- Run the API
    ```
    #Export Envs: After insderting daa into mongodb atlas cloud
    export CONNECTION_NAME=<chatbot-connection-name>

    #Now Run the FastAPI using below command and Relative Path as Chatbot-streamlit/
    uvicorn src.main:app --reload
 
Now /predict Endpoint of FastAPI is getting exposed, which can be used in out Streamlit app to do Q&A over RAG.

- Check the API working in swagger

```
    Click on the link e.g http://127.0.0.1:8000/docs to check the swagger
```

### Setup the Streamlit UI

To Run the Streamlit Application Locally Follow below Instructions to Run the App

```
    streamlit run app.py
```

This Will open a streamlit application where you can ask your questions and get the responses via API you exposed.

## Sample Output
![chatbot_img_5](https://github.com/user-attachments/assets/7f407d47-7ea6-492a-82bf-ff8d1b69b08c)
![chatbot_img_6](https://github.com/user-attachments/assets/ba6b6639-aef7-45f8-81b5-e7b9af6cf3e5)
![chatbot_img_7](https://github.com/user-attachments/assets/e23b4ede-9e60-4677-9589-d2702a15c837)
![chatbot_img_8](https://github.com/user-attachments/assets/3049bf88-6fce-4e1f-aea9-0f2897357fed)



**for more details**
Happy to Connect!! [Samiksha Kolhe](https://www.linkedin.com/in/samiksha-kolhe25701/)
