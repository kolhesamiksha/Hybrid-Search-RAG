import hashlib
import io
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import ast
import base64
from github import Github
import pandas as pd
from typing import List
import traceback

from ragas.metrics import faithfulness,context_utilization
from ragas.metrics.critique import harmfulness, correctness
from langchain_openai import ChatOpenAI
from ragas import evaluate
from langchain_core.documents import Document
from datasets import Dataset, Sequence
from ragas.llms import LangchainLLMWrapper
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from .get_insert_mongo_data import format_creds_mongo
creds_mongo = format_creds_mongo()

DENSE_EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-en"
LLM_MODEL_NAME = "llama-3.1-70b-versatile"
OPENAI_API_BASE = creds_mongo['OPENAI_API_BASE']

def decrypt_pass(base64_encoded_key):
    encrypted_key = base64.b64decode(base64_encoded_key)
    # encrypted_key = ast.literal_eval(encrypted_key)
    iv = encrypted_key[:AES.block_size]  # Extract the IV
    cipher_text = encrypted_key[AES.block_size:]  # Extract the ciphertext
    cipher = AES.new(iv, AES.MODE_CBC, iv)  # Recreate the cipher with the extracted IV
    padded_plain_text = cipher.decrypt(cipher_text)
    plain_text = unpad(padded_plain_text, AES.block_size)
    return plain_text.decode('utf-8')

GTHUB_TOKEN = decrypt_pass(creds_mongo['GITHUB_TOKEN'])
GROQ_API_KEY = decrypt_pass(creds_mongo['GROQ_API_KEY'])

def save_history_to_github(query, response):
    try:
        g = Github(GTHUB_TOKEN)
        repo = g.get_repo("kolhesamiksha/Hybrid-Search-RAG")
        contents = repo.get_contents('Chatbot-streamlit/chat_history/chat_history.csv')
        decoded_content = base64.b64decode(contents.content)
        csv_file = io.BytesIO(decoded_content)
        df = pd.read_csv(csv_file)
        new_data = pd.DataFrame({'Query': [query], 'Answer': [response[0][0]], 'Context':[response[2]]})
        concat_df = pd.concat([df, new_data], ignore_index=True)
        updated_csv = concat_df.to_csv(index=False)
        repo.update_file(contents.path, "Updated CSV File", updated_csv, contents.sha)
    except Exception as e:
        print(traceback.format_exc())

def validate_column_dtypes(ds: Dataset):
    for column_names in ["question", "answer", "ground_truth"]:
        try:
            if column_names in ds.features:
                if ds.features[column_names].feature.dtype != "string":
                    raise ValueError(
                        f'Dataset feature "{column_names}" should be of type string'
                    )
                    return "FAIL"
                else:
                    return "PASS"
        except Exception as e:
            print(traceback.format_exc())
            return "FAIL"

    for column_names in ["contexts"]:
        try:
            if column_names in ds.features:
                if not (
                    isinstance(ds.features[column_names], Sequence)
                    and ds.features[column_names].feature.dtype == "string"
                ):
                    raise ValueError(
                        f'Dataset feature "{column_names}" should be of type'
                        f" Sequence[string], got {type(ds.features[column_names])}"
                    )
                    return "FAIL"
                else:
                    return "PASS"
        except Exception as e:
            print(traceback.format_exc())
            return "FAIL"

def rag_evaluation(question:List[str], answer:List[str], context:List[List[str]]):
    try:
        data = {
            "question": question,
            "answer": answer,
            "contexts": context
        }
        rag_dataset = Dataset.from_dict(data)
        llm_chat = ChatOpenAI(
            model=LLM_MODEL_NAME,
            openai_api_base=OPENAI_API_BASE,
            openai_api_key=GROQ_API_KEY
        )
        
        evaluation_chat_model = LangchainLLMWrapper(llm_chat)
        evaluation_embeddings = FastEmbedEmbeddings(model_name=DENSE_EMBEDDING_MODEL)
        result = evaluate(rag_dataset,metrics=[faithfulness,context_utilization, harmfulness, correctness],
                            llm=evaluation_chat_model, embeddings=evaluation_embeddings)
        return result
    except Exception as e:
        print(traceback.format_exc())
        return {}