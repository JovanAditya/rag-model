#!/usr/bin/env python3
"""
API Integration Example - Academic RAG

Example of integrating Academic RAG with FastAPI for chatbot applications.
Questions are based on actual knowledge base content.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Example FastAPI integration
FASTAPI_EXAMPLE = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from rag_model import AcademicRAG

app = FastAPI(title="Academic Chatbot API")

# Initialize RAG model (do this once at startup)
rag_model = None

@app.on_event("startup")
async def startup():
    global rag_model
    rag_model = AcademicRAG()

class QueryRequest(BaseModel):
    question: str
    pipeline_type: str = "advanced"

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[dict]

@app.post("/api/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """Handle chat query."""
    try:
        result = rag_model.query(
            question=request.question,
            pipeline_type=request.pipeline_type
        )
        return QueryResponse(
            answer=result.get("answer", ""),
            confidence=result.get("confidence", 0.0),
            sources=result.get("sources", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint."""
    if rag_model:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "degraded", "model_loaded": False}
'''


def main():
    """Show API integration example."""
    print("=" * 60)
    print("Academic RAG - API Integration Example")
    print("=" * 60)
    
    print("\n## FastAPI Integration Example\n")
    print("Copy this code to create your chatbot API:\n")
    print("-" * 60)
    print(FASTAPI_EXAMPLE)
    print("-" * 60)
    
    print("\n## How to run:\n")
    print("1. Save the code above to 'chatbot_api.py'")
    print("2. Install FastAPI: pip install fastapi uvicorn")
    print("3. Run: uvicorn chatbot_api:app --reload --port 5001")
    print("4. Access API docs at: http://localhost:5001/docs")
    
    print("\n## Example API calls:\n")
    print("""
# Query - Kerja Praktek
curl -X POST "http://localhost:8000/api/chat" \\
     -H "Content-Type: application/json" \\
     -d '{"question": "Apa syarat untuk mengikuti Kerja Praktek?"}'

# Query - Tugas Akhir
curl -X POST "http://localhost:8000/api/chat" \\
     -H "Content-Type: application/json" \\
     -d '{"question": "Bagaimana prosedur pendaftaran sidang tugas akhir?"}'

# Query - RPL
curl -X POST "http://localhost:8000/api/chat" \\
     -H "Content-Type: application/json" \\
     -d '{"question": "Apa yang dimaksud dengan program RPL?"}'

# Health check
curl "http://localhost:5001/health"
""")
    
    print("=" * 60)
    print("See rag-api/ folder for complete API implementation")
    print("=" * 60)


if __name__ == "__main__":
    main()
