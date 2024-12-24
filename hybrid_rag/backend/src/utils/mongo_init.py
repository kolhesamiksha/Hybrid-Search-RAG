from pymongo import MongoClient
import os
import streamlit as st

CONNECTION_STRING = st.secrets["CONNECTION_NAME"]

def get_database(connection_string):
   client = MongoClient(connection_string)
   return client['Hybrid-search-rag']

def create_collection(dbname):
    collection_name = dbname["credentials"]
    return collection_name

def insert_data(collection_name):
    item_1 = {
    "_id" : 1,
    "cred_name" : "ZILLIZ_CLOUD_URI",
    "cred_values" : ""
    }

    item_2 = {
    "_id" : 2,
    "cred_name" : "ZILLIZ_CLOUD_API_KEY",
    "cred_values" : ""
    }
    
    item_3 ={
    "_id" : 3,
    "cred_name" : "COLLECTION_NAME",
    "cred_values" : ""
    }

    item_4 = {
    "_id" : 4,
    "cred_name" : "GROQ_API_KEY",
    "cred_values" : ""
    }

    item_5 = {
    "_id" : 5,
    "cred_name" : "GITHUB_TOKEN",
    "cred_values" : ""
    }

    item_6 = {
    "_id" : 6,
    "cred_name" : "OPENAI_API_BASE",
    "cred_values" : ""
    }
    collection_name.insert_one(item_6)                                 #insert_many([item_1,item_2,item_3,item_4])
    print("Successfully Stored Data into the Mongo")

if __name__ == "__main__":
   dbname = get_database(CONNECTION_STRING)
   collection_name = create_collection(dbname)
   insert_data(collection_name)
