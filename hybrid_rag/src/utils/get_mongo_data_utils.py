import os
from typing import Dict, List, Optional
import traceback
from hybrid_rag.src.utils.mongo_init import get_database
from hybrid_rag.src.utils.logutils import Logger

logger = Logger().get_logger()

class MongoCredentialManager:
    """
    A class to handle credential retrieval and formatting from a MongoDB collection.
    """

    def __init__(self, connection_string: str, collection_name: str, db_name: str):
        """
        Initializes the MongoCredentialManager with the connection string.

        :param connection_string: MongoDB connection string.
        :param collection_name: MongoDB collection name.
        :param db_name: MongoDB database name where data stored.
        """
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.db_name = db_name

    def __get_creds_from_mongo(self) -> List[Dict[str,str]]:
        """
        Fetches credential documents from the MongoDB collection.

        :return: Cursor object containing the documents.
        """
        try:

            dbname = get_database(self.connection_string, self.collection_name)
            collection_name = dbname[self.db_name]                             #dbname["credentials"]
            item_details = collection_name.find()
            logger.info(f"Successfully get the data from MongoDB from Collection : {self.collection_name} and DB: {self.db_name}")
            return item_details
        except Exception as e:
            error = str(e)
            logger.error(f"Failed to Retrieve the Data from MongoDB Reason: {error} -> TRACEBACK : {traceback.format_exc()}")
            raise

    def _format_creds(self) -> Dict[str,str]:
        """
        Formats the credentials retrieved from MongoDB into a dictionary.

        :return: Dictionary containing formatted credentials.
        """
        mongo_dict = {}
        try:
            item_details = self.__get_creds_from_mongo()

            for items in item_details:
                if items['cred_name'] == "GROQ_API_KEY":
                    mongo_dict['GROQ_API_KEY'] = items['cred_values']
                elif items['cred_name'] == "ZILLIZ_CLOUD_URI":
                    mongo_dict['ZILLIZ_CLOUD_URI'] = items['cred_values']
                elif items['cred_name'] == "ZILLIZ_CLOUD_API_KEY":
                    mongo_dict['ZILLIZ_CLOUD_API_KEY'] = items['cred_values']
                elif items['cred_name'] == "GITHUB_TOKEN":
                    mongo_dict['GITHUB_TOKEN'] = items['cred_values']
                elif items['cred_name'] == "COLLECTION_NAME":
                    mongo_dict['COLLECTION_NAME'] = items['cred_values']
                elif items['cred_name'] == "OPENAI_API_BASE":
                    mongo_dict['OPENAI_API_BASE'] = items['cred_values']
            logger.info(f"Successfully Created the Dictionary of MogoCreds from Collection : {self.collection_name} and DB: {self.db_name}")
        except Exception as e:
                error = str(e)
                logger.error(f"Failed to Create the Dictionary of Mongocreds Reason: {error} -> TRACEBACK : {traceback.format_exc()}")
        return mongo_dict

