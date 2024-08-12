from pydantic import BaseModel
from typing import List, Tuple

class PredictSchema(BaseModel):
    query: str
    history: List[Tuple[str, str]] = []