import os
from typing import List
from hybrid_rag.src.utils.custom_utils import SparseFastEmbedEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


def sparse_embedding_model(texts: List[str], embed_model):
    embeddings = SparseFastEmbedEmbeddings(model_name=embed_model)
    query_embeddings = embeddings.embed_documents([texts])
    return query_embeddings

def dense_embedding_model(texts: List[str], embed_model):
    embeddings = FastEmbedEmbeddings(model_name=embed_model)
    query_embeddings = embeddings.embed_documents([texts])
    return query_embeddings

def retrieval_embedding_model(model_name):
    embed_model = FastEmbedEmbeddings(model_name=model_name)
    return embed_model