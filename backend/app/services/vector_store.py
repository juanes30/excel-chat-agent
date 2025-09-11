"""Vector Store Service using ChromaDB for semantic search capabilities."""

import asyncio
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Service for managing vector embeddings and semantic search using ChromaDB."""

    def __init__(self, 
                 persist_directory: str = "chroma_db",
                 collection_name: str = "excel_documents",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the Vector Store Service.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            embedding_model: SentenceTransformer model for embeddings
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        
        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception as e:
            # Collection doesn't exist, create it
            logger.info(f"Collection not found ({e}), creating new collection: {collection_name}")
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Excel document embeddings for semantic search"}
            )
            logger.info(f"Created new collection: {collection_name}")
        
        # Cache for embeddings to avoid recomputation
        self._embedding_cache = {}

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Check cache first
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            text_hash = hash(text)
            if text_hash in self._embedding_cache:
                cached_embeddings.append((i, self._embedding_cache[text_hash]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            logger.debug(f"Generating embeddings for {len(uncached_texts)} texts")
            new_embeddings = self.embedding_model.encode(
                uncached_texts,
                convert_to_numpy=True,
                show_progress_bar=len(uncached_texts) > 10
            ).tolist()
            
            # Cache new embeddings
            for text, embedding in zip(uncached_texts, new_embeddings):
                self._embedding_cache[hash(text)] = embedding
        else:
            new_embeddings = []
        
        # Combine cached and new embeddings in correct order
        all_embeddings = [None] * len(texts)
        
        # Place cached embeddings
        for i, embedding in cached_embeddings:
            all_embeddings[i] = embedding
        
        # Place new embeddings
        for i, embedding in zip(uncached_indices, new_embeddings):
            all_embeddings[i] = embedding
        
        return all_embeddings

    async def add_excel_data(self, 
                           file_name: str,
                           file_hash: str,
                           sheets_data: Dict[str, Any],
                           batch_size: int = 50) -> bool:
        """Add Excel data to the vector store.
        
        Args:
            file_name: Name of the Excel file
            file_hash: Hash of the Excel file
            sheets_data: Dictionary containing sheet data and text chunks
            batch_size: Size of batches for processing
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if file already exists
            existing_docs = self.collection.get(
                where={"file_hash": file_hash},
                limit=1
            )
            
            if existing_docs['ids']:
                logger.info(f"File {file_name} already exists in vector store")
                return True
            
            logger.info(f"Adding Excel data for file: {file_name}")
            
            # Prepare documents for insertion
            documents = []
            metadatas = []
            ids = []
            
            for sheet_name, sheet_data in sheets_data.items():
                text_chunks = sheet_data.get('text_chunks', [])
                metadata = sheet_data.get('metadata', {})
                
                for i, chunk in enumerate(text_chunks):
                    if not chunk.strip():
                        continue
                    
                    doc_id = f"{file_hash}_{sheet_name}_{i}"
                    
                    documents.append(chunk)
                    metadatas.append({
                        "file_name": file_name,
                        "file_hash": file_hash,
                        "sheet_name": sheet_name,
                        "chunk_index": i,
                        "num_rows": metadata.get('num_rows', 0),
                        "num_cols": metadata.get('num_cols', 0),
                        "added_at": datetime.now().isoformat(),
                        "chunk_type": "data" if i > 0 else "summary"
                    })
                    ids.append(doc_id)
            
            if not documents:
                logger.warning(f"No valid documents found for file: {file_name}")
                return False
            
            # Process in batches
            total_batches = len(documents) // batch_size + (1 if len(documents) % batch_size else 0)
            
            for batch_idx in range(0, len(documents), batch_size):
                batch_end = min(batch_idx + batch_size, len(documents))
                batch_docs = documents[batch_idx:batch_end]
                batch_metas = metadatas[batch_idx:batch_end]
                batch_ids = ids[batch_idx:batch_end]
                
                # Generate embeddings for this batch
                embeddings = await asyncio.to_thread(
                    self._generate_embeddings, 
                    batch_docs
                )
                
                # Add to collection
                self.collection.add(
                    documents=batch_docs,
                    embeddings=embeddings,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
                
                logger.debug(f"Added batch {batch_idx // batch_size + 1}/{total_batches} "
                           f"({len(batch_docs)} documents)")
            
            logger.info(f"Successfully added {len(documents)} documents for {file_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding Excel data to vector store: {e}")
            return False

    async def search(self, 
                    query: str,
                    n_results: int = 5,
                    file_filter: Optional[str] = None,
                    sheet_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform semantic search in the vector store.
        
        Args:
            query: Search query
            n_results: Number of results to return
            file_filter: Optional file name filter
            sheet_filter: Optional sheet name filter
            
        Returns:
            List of search results with content and metadata
        """
        try:
            if not query.strip():
                return []
            
            # Generate query embedding
            query_embedding = await asyncio.to_thread(
                self._generate_embeddings,
                [query]
            )
            
            if not query_embedding:
                return []
            
            # Build where clause for filtering
            where_clause = {}
            if file_filter:
                where_clause["file_name"] = {"$contains": file_filter}
            if sheet_filter:
                where_clause["sheet_name"] = {"$contains": sheet_filter}
            
            # Perform search
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=min(n_results, 100),  # Limit to reasonable maximum
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            
            if results['ids'] and results['ids'][0]:  # Check if we have results
                for i in range(len(results['ids'][0])):
                    # Calculate relevance score (1 - normalized distance)
                    distance = results['distances'][0][i] if results['distances'] else 1.0
                    relevance_score = max(0, 1 - (distance / 2))  # Normalize to 0-1 range
                    
                    result = {
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "relevance_score": round(relevance_score, 4),
                        "file_name": results['metadatas'][0][i].get('file_name', ''),
                        "sheet_name": results['metadatas'][0][i].get('sheet_name', ''),
                        "chunk_index": results['metadatas'][0][i].get('chunk_index', 0)
                    }
                    formatted_results.append(result)
            
            logger.info(f"Search for '{query}' returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            return []

    async def search_by_file(self, 
                           file_name: str,
                           query: Optional[str] = None,
                           n_results: int = 10) -> List[Dict[str, Any]]:
        """Search within a specific file.
        
        Args:
            file_name: Name of the file to search in
            query: Optional search query (if None, returns all chunks)
            n_results: Number of results to return
            
        Returns:
            List of search results from the specified file
        """
        try:
            if query:
                return await self.search(
                    query=query,
                    n_results=n_results,
                    file_filter=file_name
                )
            else:
                # Return all chunks from the file
                results = self.collection.get(
                    where={"file_name": file_name},
                    limit=n_results,
                    include=["documents", "metadatas"]
                )
                
                formatted_results = []
                if results['ids']:
                    for i in range(len(results['ids'])):
                        result = {
                            "content": results['documents'][i],
                            "metadata": results['metadatas'][i],
                            "relevance_score": 1.0,  # No relevance scoring without query
                            "file_name": results['metadatas'][i].get('file_name', ''),
                            "sheet_name": results['metadatas'][i].get('sheet_name', ''),
                            "chunk_index": results['metadatas'][i].get('chunk_index', 0)
                        }
                        formatted_results.append(result)
                
                return formatted_results
                
        except Exception as e:
            logger.error(f"Error searching by file: {e}")
            return []

    async def search_by_sheet(self, 
                            file_name: str,
                            sheet_name: str,
                            query: Optional[str] = None,
                            n_results: int = 10) -> List[Dict[str, Any]]:
        """Search within a specific sheet.
        
        Args:
            file_name: Name of the file
            sheet_name: Name of the sheet
            query: Optional search query
            n_results: Number of results to return
            
        Returns:
            List of search results from the specified sheet
        """
        return await self.search(
            query=query or "",
            n_results=n_results,
            file_filter=file_name,
            sheet_filter=sheet_name
        )

    async def delete_file_data(self, file_hash: str) -> bool:
        """Delete all data for a specific file.
        
        Args:
            file_hash: Hash of the file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all document IDs for the file
            results = self.collection.get(
                where={"file_hash": file_hash},
                include=["documents"]
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} documents for file hash: {file_hash}")
                return True
            else:
                logger.warning(f"No documents found for file hash: {file_hash}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting file data: {e}")
            return False

    async def clear_all(self) -> bool:
        """Clear all data from the vector store.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete the collection
            self.client.delete_collection(name=self.collection_name)
            
            # Recreate the collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Excel document embeddings for semantic search"}
            )
            
            # Clear cache
            self._embedding_cache.clear()
            
            logger.info("Cleared all data from vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            return False

    async def reindex_all(self, excel_processor, progress_callback=None) -> bool:
        """Reindex all Excel files.
        
        Args:
            excel_processor: Excel processor instance
            progress_callback: Optional callback function for progress updates
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting full reindex")
            
            # Clear existing data
            await self.clear_all()
            
            # Get all processed files
            all_files = excel_processor.process_all_files()
            total_files = len(all_files)
            
            if progress_callback:
                await progress_callback(0, total_files, "Starting reindex...")
            
            success_count = 0
            for i, file_data in enumerate(all_files):
                try:
                    success = await self.add_excel_data(
                        file_name=file_data['file_name'],
                        file_hash=file_data['file_hash'],
                        sheets_data=file_data['sheets']
                    )
                    
                    if success:
                        success_count += 1
                    
                    if progress_callback:
                        await progress_callback(
                            i + 1, 
                            total_files, 
                            f"Processed {file_data['file_name']}"
                        )
                        
                except Exception as e:
                    logger.error(f"Error reindexing file {file_data['file_name']}: {e}")
                    continue
            
            logger.info(f"Reindex completed: {success_count}/{total_files} files processed")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error during reindex: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics.
        
        Returns:
            Dictionary with vector store statistics
        """
        try:
            # Get collection info
            collection_count = self.collection.count()
            
            # Get some sample metadata to analyze
            sample_results = self.collection.get(
                limit=min(1000, collection_count),
                include=["metadatas"]
            )
            
            # Analyze metadata
            files = set()
            sheets = set()
            chunk_types = {}
            
            if sample_results['metadatas']:
                for meta in sample_results['metadatas']:
                    if 'file_name' in meta:
                        files.add(meta['file_name'])
                    if 'sheet_name' in meta:
                        sheets.add(meta['sheet_name'])
                    
                    chunk_type = meta.get('chunk_type', 'unknown')
                    chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            return {
                "total_documents": collection_count,
                "unique_files": len(files),
                "unique_sheets": len(sheets),
                "chunk_types": chunk_types,
                "embedding_model": self.embedding_model_name,
                "collection_name": self.collection_name,
                "cache_size": len(self._embedding_cache)
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {
                "total_documents": 0,
                "unique_files": 0,
                "unique_sheets": 0,
                "chunk_types": {},
                "embedding_model": self.embedding_model_name,
                "collection_name": self.collection_name,
                "cache_size": 0
            }

    def health_check(self) -> Dict[str, Any]:
        """Check the health of the vector store service.
        
        Returns:
            Dictionary with health status
        """
        try:
            # Try to perform a simple operation
            count = self.collection.count()
            
            # Test embedding generation
            test_embedding = self._generate_embeddings(["test"])
            
            return {
                "status": "healthy",
                "document_count": count,
                "embedding_model_loaded": len(test_embedding) > 0,
                "collection_accessible": True
            }
            
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "document_count": 0,
                "embedding_model_loaded": False,
                "collection_accessible": False
            }