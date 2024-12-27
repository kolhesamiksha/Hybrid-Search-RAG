import io
import ast
import base64
from typing import List, Dict
from github import Github
import pandas as pd
from typing import List
import traceback

from hybrid_rag.src.utils.logutils import Logger

#TODO:add Logger & exceptions

def save_history_to_github(query, response, github_token):
    try:
        g = Github(github_token)
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
    

def calculate_cost(total_usage:Dict):
    # specific for gpt-4o, not generic
    completion_tokens = total_usage['token_usage']['completion_tokens']
    prompt_tokens = total_usage['token_usage']['prompt_tokens']

    #cost in $
    input_token = (prompt_tokens/1000)*0.0065
    output_token = (completion_tokens/1000)*0.0195

    total_cost = input_token+output_token
    return total_cost