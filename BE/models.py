from pydantic import BaseModel
from typing import Literal

class Intent_Classification(BaseModel):
    class_type: Literal['continue', 'new_task']

class DomainClassification(BaseModel):
    class_type: Literal['dental', 'non-dental']

class QueryClarity(BaseModel):
    class_type: Literal["FAQ", "Booking", "Vague"]

class QueryInput(BaseModel):
    user_id: str
    query: str