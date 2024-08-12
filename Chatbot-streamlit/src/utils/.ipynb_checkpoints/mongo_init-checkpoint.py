from pymongo import MongoClient
import os

CONNECTION_STRING = os.getenv('CONNECTION_STRING')

def get_database(connection_string):
   client = MongoClient(connection_string)
   return client['advance_rag_credentials']

def create_collection(dbname):
    collection_name = dbname["retrieval_creds"]
    return collection_name

def insert_data(collection_name):
    item_1 = {
    "_id" : 1,
    "cred_name" : "OPENAI_API_KEY",
    "cred_values" : ""
    }

    item_2 = {
    "_id" : 2,
    "cred_name" : "OPENAI_API_BASE",
    "cred_values" : ""
    }

    item_3 = {
    "_id" : 3,
    "cred_name" : "ZILLIZ_CLOUD_URI",
    "cred_values" : ""
    }

    item_4 = {
    "_id" : 4,
    "cred_name" : "ZILLIZ_CLOUD_API_KEY",
    "cred_values" : ""
    }
    
    item_5 ={
    "_id" : 5,
    "cred_name" : "COLLECTION_NAME",
    "cred_values" : "advance_rag_sam"
    }

    collection_name.insert_one(item_5)                                 #insert_many([item_1,item_2,item_3,item_4])
    print("Successfully Stored Data into the Mongo")

if __name__ == "__main__":
   dbname = get_database(CONNECTION_STRING)
   collection_name = create_collection(dbname)
   insert_data(collection_name)