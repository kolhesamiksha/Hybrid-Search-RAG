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
from hybrid_rag.src.models import LLMModelInitializer
from hybrid_rag.src.prompts.followup_prompt import (
    FollowupPromptGenerator,
)

# Custom Query Expansion Class
class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        cleaned_lines = [re.sub(r"^\d+\.\s*", "", line) for line in lines]
        return cleaned_lines

class FollowupQGeneration:
    def __init__(self, llm_model:LLMModelInitializer, followup_template: FollowupPromptGenerator, logger: Optional[logging.Logger] = None):
        self.llm_model = llm_model.initialise_llm_model()
        self.FOLLOWUP_PROMPT_TEMPLATE = followup_template.generate_prompt()
        self.logger = logger

    async def generate_followups(self, question: str, context: List[Document], response:str) -> List[str]:
        try:

            #Rnnable Parallel Chaining reduces the chaining time by 30%
            output_parser = LineListOutputParser()
            chain = (
                RunnableParallel(
                    {
                        "response": RunnablePassthrough(),
                        "context": RunnablePassthrough(),
                        "question": RunnablePassthrough(),
                    }
                )
                | self.FOLLOWUP_PROMPT_TEMPLATE 
                | self.llm_model 
                | output_parser
            )

            result = await chain.ainvoke({"question": question, "context": context, "response": response})
            self.logger.info("Successfully Generated the Followup Questions!")
            return result
        except Exception as e:
            self.logger.info(f"Error while Generating the Followup question : {str(e)} traceback: {traceback.format_exc()}")
            raise


