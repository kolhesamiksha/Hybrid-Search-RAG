import logging
import ast
import traceback

from typing import Optional
from typing import List
from typing import Dict
from hybrid_rag.src.utils.logutils import Logger
from pymilvus import MilvusClient
import time


class MilvusMetaFiltering:
    """
    A class to interact with a Milvus collection and query documents based on a filter expression.

    Attributes:
        uri (str): The URI for the Milvus instance.
        token (str): The authentication token for the Milvus instance.
        collection_name (str): The name of the Milvus collection to query.
        client (MilvusClient): Instance of the MilvusClient for performing operations.
    """

    def __init__(self, zillinz_cloud_uri: str, zillinz_cloud_api_key: str, collection_name: str, metadata_attributes: List[Dict[str,str]], logger: Optional[logging.Logger] = None):
        """
        Initializes the MilvusQueryHandler with the specified URI, token, and collection name.

        Args:
            uri (str): The URI for the Milvus instance.
            token (str): The authentication token for the Milvus instance.
            collection_name (str): The name of the Milvus collection to query.
        """
        self.logger = logger if logger else Logger().get_logger()
        self.uri = zillinz_cloud_uri
        self.token = zillinz_cloud_api_key
        self.collection_name = collection_name
        self.client = MilvusClient(uri=self.uri, token=self.token)
        self.metadata_attributes_lst = self._create_metadata_list(metadata_attributes)

    def prepare_metadata_expression(self, meta_field: str, topics_extracted: List[str]) -> str:
        """
        Prepare metadata expression for filtering documents.

        Args:
            meta_field (str): The metadata field to filter on.
            topics_extracted (List[str]): List of extracted topics for filtering.

        Returns:
            str: Metadata expression string.
        """
        if isinstance(topics_extracted, str):
            topics_extracted = ast.literal_eval(topics_extracted)

        if not isinstance(topics_extracted, list):
            raise ValueError("topics_extracted must be a list or a string representation of a list.")
        
        filter_expr = " or ".join(
            [f"{meta_field} like '%{topic.strip()}%'" for topic in topics_extracted]
        )
        self.logger.info(f"Successfully Generated the Filter Expression: {filter_expr}")
        return filter_expr
    
    def _create_metadata_list(self, metadata_attributes: List[Dict[str,str]]) -> List[str]:
        attribute_info = []

        for i, meta in enumerate(metadata_attributes):
            attribute_name = meta.get(f'METADATA_ATTRIBUTE{i+1}_NAME', '').strip()
            if attribute_name:
                attribute_info.append(attribute_name)

        return attribute_info
    
    def _process_results(self, hits: List[dict]):
        """
        Process the search results and create a list of Document objects.

        Args:
            res (Any): The search results from Milvus, could be a list, dictionary, or another format depending on the Milvus client.

        Returns:
            List[Document]: A list of Document objects containing page content and metadata.
        """
        for hit in hits:
            page_content = hit.get("text", "text")
            metadata = {attr: hit.get(attr, attr) for attr in self.metadata_attributes_lst}
            #doc_chunk = Document(page_content=page_content, metadata=metadata)
            yield {
                "page_content": page_content,
                "metadata": metadata
            }

    def query_documents(self, filter_expr: str):
        """
        Queries the Milvus collection based on a filter expression.

        Args:
            filter_expr (str): The filter expression to apply to the query.
            output_fields (list): A list of fields to retrieve in the query results.

        Returns:
            list: A list of documents matching the filter expression.
        """
        try:
            start_time = time.time()
            
            # Perform the query
            hits = self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=["text"] + self.metadata_attributes_lst,
            )
            self.logger.info(f"Milvus Meta Filtering Documents Extracted successfully in {time.time() - start_time:.2f} seconds.")
            self.logger.info(f"Number of Documents Retrieved during Metadata Filtering for Summarization: {len(hits)}")
            results = self._process_results(hits)
            for res in results:
                yield res
        except Exception as e:
            self.logger.error(f"An error occurred while querying Milvus: {traceback.format_exc()}")
            raise