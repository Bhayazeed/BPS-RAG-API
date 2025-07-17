from pydantic import BaseModel, Field
from typing import List

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length = 500, description="Pertanyaan yang akan diajukan ke RAG Chain.")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="Jawaban yang dihasilkan oleh RAG chain.")

class IngestResponse(BaseModel):
    filename: str
    message: str
    chunks_created: int

class DocumentListResponse(BaseModel):
    processed_documents: List[str]

class DebugQueryResponse(BaseModel):
    question: str
    retrieved_context: str
    answer: str
    