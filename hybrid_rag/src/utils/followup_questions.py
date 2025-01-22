"""
Module Name: hybrid_search.py
Author: Samiksha Kolhe
Version: 0.1.0
"""

#question & response 

#summarization support

#guardrails support: ingrained inside the module

import re
import logging
import traceback
from typing import Optional, List

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough
)
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from hybrid_rag.src.utils.logutils import Logger


# Custom Query Expansion Class
class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        cleaned_lines = [re.sub(r"^\d+\.\s*", "", line) for line in lines]
        return cleaned_lines

class FollowupQGeneration:
    def __init__(self, llm_model, followup_template, logger: Optional[logging.Logger] = None):
        self.llm_model = llm_model
        self.FOLLOWUP_PROMPT_TEMPLATE = followup_template
        self.logger = logger

    def generate_followups(self, context: List[Document], response:str) -> List[str]:

        try:
            runner = RunnableParallel(
                {"response": response, "context": context, "question": RunnablePassthrough()}
            )

            output_parser = LineListOutputParser()
            output = runner | self.FOLLOWUP_PROMPT_TEMPLATE | self.llm_model | output_parser
            self.logger.info("Successfully Generated the Followup Questions!")
        except Exception as e:
            output = str(e)
            self.logger.info(f"Error while Generating the Followup question : {str(e)} traceback: {traceback.format_exc()}")
        
        return output


