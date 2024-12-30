"""
Module Name: hybrid_search.py
Author: Samiksha Kolhe
Version: 0.1.0
"""
from pymongo import MongoClient
from pymongo.collection import Collection


class MongoDBHandler:
    def __init__(self, connection_string: str, db_name: str):
        """
        Initialize the MongoDBHandler class.

        :param connection_string: MongoDB connection string.
        :param db_name: Name of the default database to connect to.
        """
        self.connection_string = connection_string
        self.db_name = db_name
        self.client: MongoClient = MongoClient(self.connection_string)
        self.db: MongoClient = self.client[self.db_name]

    def get_or_create_database(self, db_name: str) -> MongoClient:
        """
        Switch to a different database.

        :param db_name: Name of the database to switch to.
        :return: The database object.
        """
        self.db = self.client[db_name]
        return self.db

    def get_or_create_collection(self, collection_name: str) -> Collection:
        """
        Create or get a MongoDB collection.

        :param collection_name: Name of the collection to create or fetch.
        :return: The collection object.
        """
        return self.db[collection_name]

    def insert_data(self, collection_name: str, data: list[dict]) -> None:
        """
        Insert data into a specified collection.

        :param collection_name: Name of the collection.
        :param data: List of dictionaries to insert into the collection.
        """
        collection = self.get_or_create_collection(collection_name)
        try:
            if isinstance(data, list):
                collection.insert_many(data)
            elif isinstance(data, dict):
                collection.insert_one(data)
            else:
                raise ValueError("Data must be a dictionary or a list of dictionaries.")
            print("Successfully stored data into MongoDB.")
        except Exception as e:
            print(f"Error inserting data: {e}")

    def close_connection(self) -> None:
        """
        Close the MongoDB connection.
        """
        self.client.close()


# Example usage:
# if __name__ == "__main__":
#     CONNECTION_STRING = "your_mongo_connection_string_here"
#     DB_NAME = "your_db_name"
#     COLLECTION_NAME = "your_collection_name"

#     data_to_insert = [
#         {"_id": 1, "cred_name": "ZILLIZ_CLOUD_URI", "cred_values": ""},
#         {"_id": 2, "cred_name": "ZILLIZ_CLOUD_API_KEY", "cred_values": ""},
#         {"_id": 3, "cred_name": "COLLECTION_NAME", "cred_values": ""},
#         {"_id": 4, "cred_name": "GROQ_API_KEY", "cred_values": ""},
#         {"_id": 5, "cred_name": "GITHUB_TOKEN", "cred_values": ""},
#         {"_id": 6, "cred_name": "OPENAI_API_BASE", "cred_values": ""},
#     ]

#     # Create an instance of MongoDBHandler
#     mongo_handler = MongoDBHandler(CONNECTION_STRING, DB_NAME)

#     # Insert data into the collection
#     mongo_handler.insert_data(COLLECTION_NAME, data_to_insert)

#     # Switch to another database and perform operations
#     new_db = mongo_handler.get_database("NewDatabaseName")
#     print(f"Switched to database: {new_db.name}")

#     # Close the connection
#     mongo_handler.close_connection()
