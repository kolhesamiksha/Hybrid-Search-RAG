from .mongo_init import get_database
import os
import streamlit as st

CONNECTION_STRING = st.secrets["CONNECTION_NAME"]

def get_creds_from_mongo(connection_string):
    dbname = get_database(connection_string)
    collection_name = dbname["credentials"]
    item_details = collection_name.find()
    return item_details

def format_creds_mongo():
    mongo_dict = {}
    item_details = get_creds_from_mongo(CONNECTION_STRING)
    for items in item_details:
        if items['cred_name']=="GROQ_API_KEY":
            mongo_dict['GROQ_API_KEY'] = items['cred_values']
        if items['cred_name']=="ZILLIZ_CLOUD_URI":
            mongo_dict['ZILLIZ_CLOUD_URI'] = items['cred_values']
        if items['cred_name']=="ZILLIZ_CLOUD_API_KEY":
            mongo_dict['ZILLIZ_CLOUD_API_KEY'] = items['cred_values']
        if items['cred_name']=="ZILLIZ_CLOUD_API_KEY":
            mongo_dict['ZILLIZ_CLOUD_API_KEY'] = items['cred_values']
        if items['cred_name']=="GITHUB_TOKEN":
            mongo_dict['GITHUB_TOKEN'] = items['cred_values']
        if items['cred_name']=="COLLECTION_NAME":
            mongo_dict['COLLECTION_NAME'] = items['cred_values']
        if items['cred_name']=="OPENAI_API_BASE":
            mongo_dict['OPENAI_API_BASE'] = items['cred_values']
    return mongo_dict
