from typing import List
from typing import Tuple

from pydantic import BaseModel
from pydantic import Field


class ResponseSchema(BaseModel):
    query: str = Field(default=None)
    history: List[Tuple[str, str]] = []
