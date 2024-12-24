from pydantic import BaseModel
from typing import List, Tuple

class ResponseSchema(BaseModel):
    query: str
    history: List[Tuple[str, str]] = []