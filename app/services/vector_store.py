import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import CHROMA_DB_DIR, EMBEDDING_MODEL
from app.models.schemas import DocumentMetadata, DocumentChunk, VectorStoreResponse, FileType


class VectorStore:
    def __init__(self, collection_name: str = "documents"):
        """Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection to use.
        """
        self.collection_name = collection_name
        self.db_path = Path(CHROMA_DB_DIR)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize the embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Initialize the vector store
        self.db = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.db_path)
        )
    
    def store_document(self, chunks: List[DocumentChunk], metadata: Dict[str, Any]) -> VectorStoreResponse:
        try:
            # Validate metadata
            if "filename" not in metadata:
                raise ValueError("Required metadata field 'filename' is missing")
            
            # Store all chunks in the vector database
            texts = []
            metadatas = []
            ids = []
            
            for chunk in chunks:
                # Prepare metadata for Chroma (flatten it)
                chroma_metadata = {
                    "doc_id": chunk.id,
                    "filename": chunk.metadata.filename,
                    "content_length": len(chunk.text),
                    "upload_timestamp": chunk.metadata.upload_timestamp,
                    "file_type": chunk.metadata.file_type.value
                }
                
                # Add any additional simple metadata fields
                if chunk.metadata.additional_metadata:
                    for key, value in chunk.metadata.additional_metadata.items():
                        # Only include simple types
                        if isinstance(value, (str, int, float, bool)):
                            chroma_metadata[key] = value
                
                texts.append(chunk.text)
                metadatas.append(chroma_metadata)
                ids.append(chunk.id)
            
            # Add all chunks to vector store
            self.db.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            return VectorStoreResponse(
                success=True,
                message=f"Document added successfully and split into {len(chunks)} chunks",
                document_ids=[chunk.id for chunk in chunks]
            )
        except Exception as e:
            return VectorStoreResponse(
                success=False,
                message="Failed to add document to vector store",
                error=str(e)
            )
    
    def search_similar(self, query_text: str, k: int = 5) -> List[DocumentChunk]:
        results = self.db.similarity_search_with_relevance_scores(query_text, k=k)
        
        # Convert results to DocumentChunk objects
        documents = []
        for doc, score in results:
            # Extract metadata from the document
            metadata = doc.metadata
            doc_id = metadata.get("doc_id", str(uuid.uuid4()))
            
            # Ensure file_type is a proper FileType enum
            file_type = metadata.get("file_type")
            if isinstance(file_type, str):
                try:
                    file_type = FileType(file_type.lower())
                except ValueError:
                    # Default to TXT if invalid
                    file_type = FileType.TXT
            elif not isinstance(file_type, FileType):
                file_type = FileType.TXT
            
            # Format metadata to match DocumentMetadata schema
            document_metadata = DocumentMetadata(
                filename=metadata.get("filename", "unknown.txt"),
                file_type=file_type,
                content_length=metadata.get("content_length", len(doc.page_content)),
                upload_timestamp=metadata.get("upload_timestamp", ""),
                additional_metadata=metadata
            )
            
            # Create document chunk
            chunk = DocumentChunk(
                id=doc_id,
                text=doc.page_content,
                metadata=document_metadata,
                embedding_id=doc_id
            )
            
            documents.append(chunk)
        
        return documents
    
    def delete_document(self, doc_id: str) -> VectorStoreResponse:
        try:
            self.db.delete([doc_id])
            return VectorStoreResponse(
                success=True,
                message=f"Document {doc_id} deleted successfully"
            )
        except Exception as e:
            return VectorStoreResponse(
                success=False,
                message=f"Failed to delete document {doc_id}",
                error=str(e)
            )
    
    def get_document(self, doc_id: str) -> Optional[DocumentChunk]:
        results = self.db.similarity_search(
            query="",  # Empty query to bypass similarity search
            k=100,     # Retrieve a large number to increase chances of finding the document
            filter={"doc_id": doc_id}  # Filter by document ID
        )
        
        for doc in results:
            metadata = doc.metadata
            if metadata.get("doc_id") == doc_id:
                # Ensure file_type is a proper FileType enum
                file_type = metadata.get("file_type")
                if isinstance(file_type, str):
                    try:
                        file_type = FileType(file_type.lower())
                    except ValueError:
                        # Default to TXT if invalid
                        file_type = FileType.TXT
                elif not isinstance(file_type, FileType):
                    file_type = FileType.TXT
                
                # Format metadata to match DocumentMetadata schema
                document_metadata = DocumentMetadata(
                    filename=metadata.get("filename", "unknown.txt"),
                    file_type=file_type,
                    content_length=metadata.get("content_length", len(doc.page_content)),
                    upload_timestamp=metadata.get("upload_timestamp", ""),
                    additional_metadata=metadata
                )
                
                return DocumentChunk(
                    id=doc_id,
                    text=doc.page_content,
                    metadata=document_metadata,
                    embedding_id=doc_id
                )
        
        return None 