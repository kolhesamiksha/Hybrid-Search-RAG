"""
Module Name: hybrid_search.py
Author: Samiksha Kolhe
Version: 0.1.0
"""
import base64
import io
import traceback
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd
from github import Github

# TODO:add Logger & exceptions


def save_history_to_github(query:str, response:Tuple[str, float, List[Any], Dict[Any], Dict[Any]], github_token:str, repo_name:str, chatfile_name:str) -> None:
    try:
        g = Github(github_token)
        repo = g.get_repo(repo_name)
        contents = repo.get_contents(chatfile_name)
        decoded_content = base64.b64decode(contents.content)
        csv_file = io.BytesIO(decoded_content)
        df = pd.read_csv(csv_file)
        new_data = pd.DataFrame(
            {"Query": [query], "Answer": [response[0][0]], "Context": [response[2]]}
        )
        concat_df = pd.concat([df, new_data], ignore_index=True)
        updated_csv = concat_df.to_csv(index=False)
        repo.update_file(contents.path, "Updated CSV File", updated_csv, contents.sha)
    except Exception:
        print(traceback.format_exc())


def calculate_cost(total_usage: Dict) -> float:
    # specific for gpt-4o, not generic
    completion_tokens = total_usage["token_usage"]["completion_tokens"]
    prompt_tokens = total_usage["token_usage"]["prompt_tokens"]

    # cost in $
    input_token = (prompt_tokens / 1000) * 0.0065
    output_token = (completion_tokens / 1000) * 0.0195

    total_cost = input_token + output_token
    return total_cost
