"""
Module Name: hybrid_search.py
Author: Samiksha Kolhe
Version: 0.1.0
"""
import base64
import io
import os
import traceback
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import boto3
import pandas as pd
from github import Github

# TODO:add Logger & exceptions


def save_history_to_github(
    query: str,
    response: Tuple[str, float, List[Any], Dict[Any, Any], Dict[Any, Any]],
    github_token: str,
    repo_name: str,
    chatfile_name: str,
) -> None:
    try:
        g = Github(github_token)
        repo = g.get_repo(repo_name)
        contents = repo.get_contents(chatfile_name)
        decoded_content = base64.b64decode(contents.content)
        csv_file = io.BytesIO(decoded_content)
        df = pd.read_csv(csv_file)
        if not response[3]:
            faithfullness = "NULL"
            answer_relevancy = "NULL"
            context_precision = "NULL"
        else:
            faithfullness = response[3][0]["faithfullness"]
            answer_relevancy = response[3][0]["answer-relevancy"]
            context_precision = response[3][1]

        new_data = pd.DataFrame(
            {
                "Query": [query],
                "Answer": [response[0]],
                "Context": [response[2]],
                "Faithfullness": [faithfullness],
                "Answer-Relevancy": [answer_relevancy],
                "Context-Precision": [context_precision],
                "Latency": [response[1]],  # Example placeholder, adjust as needed
                "Total Cost": [response[4]],  # Example placeholder, adjust as needed
            }
        )
        concat_df = pd.concat([df, new_data], ignore_index=True)
        updated_csv = concat_df.to_csv(index=False)
        repo.update_file(contents.path, "Updated CSV File", updated_csv, contents.sha)
    except Exception:
        print(traceback.format_exc())


def save_history_to_s3(
    query: str,
    response: Tuple[str, float, List[Any], Dict[Any, Any], Dict[Any, Any]],
    s3_bucket: str,
    s3_key: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
) -> None:
    try:
        # Initialize S3 client
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

        # Fetch the existing CSV file from S3
        try:
            s3_object = s3.get_object(Bucket=s3_bucket, Key=s3_key)
            csv_data = s3_object["Body"].read()
            df = pd.read_csv(io.BytesIO(csv_data))
        except s3.exceptions.NoSuchKey:
            print("Inside Exception")
            # If the file doesn't exist, create an empty DataFrame
            df = pd.DataFrame(
                columns=[
                    "Query",
                    "Answer",
                    "Context",
                    "Faithfullness",
                    "Answer-Relevancy",
                    "Context-Precision",
                    "Latency",
                    "Total Cost",
                ]
            )

        if not response[3]:
            faithfullness = "NULL"
            answer_relevancy = "NULL"
            context_precision = "NULL"
        else:
            faithfullness = response[3][0]["faithfullness"]
            answer_relevancy = response[3][0]["answer-relevancy"]
            context_precision = response[3][1]
        # Prepare new data to append
        new_data = pd.DataFrame(
            {
                "Query": [query],
                "Answer": [response[0]],
                "Context": [response[2]],
                "Faithfullness": [faithfullness],
                "Answer-Relevancy": [answer_relevancy],
                "Context-Precision": [context_precision],
                "Latency": [response[1]],  # Example placeholder, adjust as needed
                "Total Cost": [response[4]],  # Example placeholder, adjust as needed
            }
        )

        # Concatenate the new data with the existing dataframe
        concat_df = pd.concat([df, new_data], ignore_index=True)

        # Convert the updated DataFrame back to CSV format
        updated_csv = concat_df.to_csv(index=False)

        # Upload the updated CSV to S3
        s3.put_object(Bucket=s3_bucket, Key=s3_key, Body=updated_csv)

        print("History saved to S3 successfully.")

    except Exception as e:
        print(f"Error occurred while saving to S3: {e}")
        print(f"ERROR: str(e) TRACEBACK: {traceback.format_exc()}")


def calculate_cost_openai(total_usage: Dict) -> float:
    # specific for gpt-4o, not generic
    completion_tokens = total_usage["token_usage"]["completion_tokens"]
    prompt_tokens = total_usage["token_usage"]["prompt_tokens"]

    # cost in $
    input_token = (prompt_tokens / 1000) * 0.0065
    output_token = (completion_tokens / 1000) * 0.0195

    total_cost = input_token + output_token
    return total_cost


def calculate_cost_groq_llama31(
    total_usage: Dict,
    input_token_price_per_million: float = 0.0025,
    output_token_price_per_million: float = 0.0064,
) -> float:  # Default: $0.05 per 20 million input tokens, so price per million
    """
    For llama3.1-8b-instant
    Calculate the cost based on the usage of input and output tokens.

    Pricing:
    - Input tokens: $0.05 per 20M tokens ($0.0000025 per token)
    - Output tokens: $0.08 per 12.5M tokens ($0.0000064 per token)

    :param total_usage: A dictionary containing token usage details.
        Example: {"token_usage": {"completion_tokens": int, "prompt_tokens": int}}
    :return: Total cost in dollars (float).
    """
    completion_tokens = total_usage["token_usage"]["completion_tokens"]
    prompt_tokens = total_usage["token_usage"]["prompt_tokens"]
    print(type(input_token_price_per_million))
    print(type(input_token_price_per_million))
    # Calculate token costs
    input_token_cost = (prompt_tokens / 1_000_000) * float(
        input_token_price_per_million
    )  # $ per token for input
    output_token_cost = (completion_tokens / 1_000_000) * float(
        output_token_price_per_million
    )  # $ per token for output

    # Total cost
    total_cost = input_token_cost + output_token_cost
    return total_cost
