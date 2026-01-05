from pydantic import BaseModel, Field
from typing import List, Optional, Union


class Metadata(BaseModel):
    Summary: List[str] = Field(default_factory=list, description="Summary of the documents")
    Title: str
    Author: List[str]
    DateCreated: str
    LastModifiedDate: str
    Publisher: str
    Language: str
    PageCount: Union[int, str]  # Can be "Not Available"
    SentimentTone: str