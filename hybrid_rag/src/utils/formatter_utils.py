from typing import Tuple

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts.prompt import PromptTemplate


class DocumentFormatter:
    """
    A class for formatting documents and AIMessage results into structured outputs.
    """

    @staticmethod
    def format_document(doc: Document) -> str:
        """
        Format a single document using a specified prompt template.

        :param doc: Document object to format.
        :return: Formatted string representation of the document.
        :raises ValueError: If required metadata fields are missing.
        """
        prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}"
        )

        if "source_link" in doc.metadata.keys():
            prompt += PromptTemplate(
                input_variables=["source_link"], template="\n[Source: {source_link}]"
            )

        base_info = {"page_content": doc.page_content, **doc.metadata}
        missing_metadata = set(prompt.input_variables).difference(base_info)

        if len(missing_metadata) > 0:
            required_metadata = [
                iv for iv in prompt.input_variables if iv != "page_content"
            ]
            raise ValueError(
                f"Document prompt requires documents to have metadata variables: "
                f"{required_metadata}. Received document with missing metadata: "
                f"{list(missing_metadata)}."
            )

        return prompt.format(**base_info)

    @staticmethod
    def format_docs(docs: list[Document]) -> str:
        """
        Format a list of documents into a structured string.

        :param docs: List of Document objects to format.
        :return: Formatted string representation of the documents.
        """
        return "\n\n".join(DocumentFormatter.format_document(doc) for doc in docs)

    @staticmethod
    def format_result(result: AIMessage) -> Tuple[str, int]:
        """
        Extract and return the response content and metadata from an AIMessage.

        :param result: AIMessage object.
        :return: Tuple of (response_content, response_metadata).
        """
        response = result.content
        response_metadata = result.response_metadata
        return response, response_metadata
