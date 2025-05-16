from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Any

from app.services.vector_store import VectorStore
from app.models.schemas import StoreDocumentResponse, DocumentChunk, VectorStoreResponse


router = APIRouter(prefix="/api/v1")

def get_vector_store():
    return VectorStore()


@router.post("/store", response_model=StoreDocumentResponse)
async def store_document(
    chunks: List[DocumentChunk],
    metadata: Dict[str, Any],
    vector_store: VectorStore = Depends(get_vector_store)
):
    response = vector_store.store_document(chunks, metadata)
    return StoreDocumentResponse(
        success=response.success,
        message=response.message,
        document_ids=response.document_ids,
        error=response.error,
        document_count=len(chunks) if response.success else 0
    )


@router.get("/documents/{doc_id}", response_model=DocumentChunk)
async def get_document(
    doc_id: str,
    vector_store: VectorStore = Depends(get_vector_store)
):
    document = vector_store.get_document(doc_id)
    
    if not document:
        raise HTTPException(status_code=404, detail=f"Document with ID {doc_id} not found")
    
    return document


@router.delete("/documents/{doc_id}", response_model=VectorStoreResponse)
async def delete_document(
    doc_id: str,
    vector_store: VectorStore = Depends(get_vector_store)
):
    return vector_store.delete_document(doc_id)


@router.get("/search", response_model=List[DocumentChunk])
async def search_documents(
    query: str = Query(..., description="The search query"),
    limit: int = Query(5, description="Maximum number of results to return"),
    vector_store: VectorStore = Depends(get_vector_store)
):
    return vector_store.search_similar(query, k=limit) 