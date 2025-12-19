"""ChromaDB vector store for Advanced RAG Pipeline."""

import os
import logging
from typing import List, Dict, Any, Optional, Union
import uuid
import numpy as np
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError:
    logging.error("ChromaDB not available. Install with: pip install chromadb")
    raise

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logging.error("sentence-transformers not available. Install with: pip install sentence-transformers")
    raise

from ..core.config import EmbeddingConfig, IndexConfig
from ..utils.logging import PipelineLogger
from ..utils.helpers import ensure_directory_exists


class IndobertEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """
    Custom embedding function using IndoBERT for ChromaDB.
    Properly handles non-sentence-transformers models like IndoBERT.
    """

    def __init__(self, model_name: str = "indobenchmark/indobert-base-p2", device: str = "cpu"):
        """
        Initialize embedding function.

        Args:
            model_name: Name of the embedding model
            device: Device to run inference on
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.logger = logging.getLogger(__name__)
        self.use_transformers = False  # Flag untuk IndoBERT

    def _load_model(self):
        """Load the appropriate model based on model name."""
        self.logger.info(f"Loading embedding model: {self.model_name}")
        
        # Cek apakah ini IndoBERT yang perlu handling khusus
        if "indobenchmark" in self.model_name.lower() or "indobert" in self.model_name.lower():
            # Gunakan transformers langsung untuk IndoBERT
            try:
                from transformers import AutoTokenizer, AutoModel, AutoConfig
                import torch
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                
                # Load model dengan cara yang menghindari meta tensor issue
                try:
                    # Coba load normal terlebih dahulu tanpa accelerate features
                    self.model = AutoModel.from_pretrained(
                        self.model_name,
                        low_cpu_mem_usage=False,
                        device_map=None,
                        dtype=torch.float32,
                        _fast_init=False  # Disable fast init yang menyebabkan meta tensor
                    )
                except Exception as e1:
                    self.logger.warning(f"Normal load failed ({e1}), trying alternative method...")
                    # Alternative: Load config first, then weights
                    config = AutoConfig.from_pretrained(self.model_name)
                    self.model = AutoModel.from_config(config)
                    # Load state dict
                    from transformers.utils import cached_file
                    import safetensors.torch
                    try:
                        weights_file = cached_file(self.model_name, "model.safetensors")
                        state_dict = safetensors.torch.load_file(weights_file)
                    except:
                        weights_file = cached_file(self.model_name, "pytorch_model.bin")
                        state_dict = torch.load(weights_file, map_location="cpu")
                    self.model.load_state_dict(state_dict)
                
                self.model = self.model.to("cpu")
                self.model.eval()
                
                if self.device == "cuda" and torch.cuda.is_available():
                    self.model = self.model.to("cuda")
                
                self.use_transformers = True
                self.logger.info("IndoBERT loaded successfully using transformers library")
            except Exception as e:
                self.logger.error(f"Failed to load IndoBERT: {e}")
                raise RuntimeError(f"IndoBERT loading failed: {e}. Please try clearing the model cache at ~/.cache/huggingface/hub/")
        else:
            # Gunakan sentence-transformers untuk model lain
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.use_transformers = False
            self.logger.info("Model loaded using sentence-transformers")

    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embedding."""
        import torch
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for input texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if self.model is None:
            self._load_model()

        if self.use_transformers:
            # IndoBERT dengan transformers
            import torch
            
            encoded = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            )
            
            if self.device == "cuda" and torch.cuda.is_available():
                encoded = {k: v.to("cuda") for k, v in encoded.items()}
            
            with torch.no_grad():
                outputs = self.model(**encoded)
            
            embeddings = self._mean_pooling(outputs, encoded['attention_mask'])
            return embeddings.cpu().numpy().tolist()
        else:
            # Sentence-transformers
            embeddings = self.model.encode(texts)
            return embeddings.tolist() if hasattr(embeddings, 'tolist') else list(embeddings)


class VectorStore:
    """
    Manage vector embeddings storage and retrieval using ChromaDB with IndoBERT embeddings.

    Provides persistent storage, efficient similarity search, and metadata filtering.
    """

    def __init__(
        self,
        collection_name: str,
        embedding_config: Optional[EmbeddingConfig] = None,
        index_config: Optional[IndexConfig] = None,
        logger: Optional[PipelineLogger] = None
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_config: Configuration for embedding model
            index_config: Configuration for indexes (ChromaDB and BM25)
            logger: Optional logger instance
        """
        self.collection_name = collection_name
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.index_config = index_config or IndexConfig()
        self.logger = logger or logging.getLogger(__name__)

        # Ensure persist directory exists
        ensure_directory_exists(self.index_config.chroma_dir)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.index_config.chroma_dir
        )

        # Initialize embedding function
        self.embedding_function = IndobertEmbeddingFunction(
            model_name=self.embedding_config.model_name,
            device=self.embedding_config.device
        )

        # Get or create collection
        self.collection = self._get_or_create_collection()

        self.logger.info(f"VectorStore initialized: {collection_name}")
        self.logger.info(f"Persist directory: {self.index_config.chroma_dir}")

    def _get_or_create_collection(self):
        """
        Get existing collection or create new one.

        Returns:
            ChromaDB collection
        """
        try:
            # Try to get existing collection
            collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            self.logger.info(f"Found existing collection: {self.collection_name}")
            return collection
        except Exception:
            # Create new collection
            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            self.logger.info(f"Created new collection: {self.collection_name}")
            return collection

    def add_documents(
        self,
        chunks: List[Dict[str, Any]]
    ) -> None:
        """
        Embed and store document chunks in ChromaDB.

        Args:
            chunks: List of document chunks with text and metadata
        """
        if not chunks:
            self.logger.warning("No chunks to add")
            return

        self.logger.info(f"Adding {len(chunks)} chunks to vector store")

        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            text = chunk.get('text', '').strip()
            if not text:
                continue

            # Generate unique ID
            chunk_id = str(uuid.uuid4())
            ids.append(chunk_id)

            # Store document text
            documents.append(text)

            # Prepare metadata (exclude large fields and convert to JSON serializable)
            metadata = chunk.get('metadata', {}).copy()

            # Remove non-serializable fields
            for key, value in metadata.items():
                if isinstance(value, (list, dict)):
                    try:
                        # Convert to string if it's complex
                        metadata[key] = str(value)
                    except Exception:
                        metadata[key] = None
                elif not isinstance(value, (str, int, float, bool, type(None))):
                    metadata[key] = str(value)

            # Add source and chunk_id to metadata (preserve existing if present)
            if 'source' not in metadata:
                metadata['source'] = chunk.get('metadata', {}).get('source', '')
            if 'chunk_id' not in metadata or not metadata['chunk_id']:
                metadata['chunk_id'] = chunk.get('metadata', {}).get('chunk_id', '')

            metadatas.append(metadata)

        if not documents:
            self.logger.warning("No valid documents to add after filtering")
            return

        # Add to collection in batches to avoid memory issues
        batch_size = self.embedding_config.batch_size
        total_added = 0
        consecutive_failures = 0
        max_consecutive_failures = 3  # Stop after 3 consecutive failures

        for i in range(0, len(documents), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_documents = documents[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]

            try:
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
                total_added += len(batch_documents)
                consecutive_failures = 0  # Reset on success

                if (i + batch_size) % 100 == 0:  # Progress update every 100 chunks
                    self.logger.info(f"Added {total_added}/{len(documents)} chunks")

            except Exception as e:
                consecutive_failures += 1
                self.logger.error(f"Failed to add batch {i//batch_size}: {e}")
                
                if consecutive_failures >= max_consecutive_failures:
                    self.logger.error(f"Stopping after {max_consecutive_failures} consecutive failures")
                    break
                continue

        self.logger.info(f"Successfully added {total_added} chunks to vector store")
        return total_added  # Return count for caller to check

    def similarity_search(
        self,
        query: str,
        k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most similar chunks using cosine similarity.

        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filters

        Returns:
            List of retrieved documents with scores and metadata
        """
        self.logger.info(f"Searching for top-{k} documents for query: {query[:50]}...")

        try:
            # Prepare query arguments
            query_args = {
                "query_texts": [query],
                "n_results": k
            }

            # Add filter if provided
            if filter_dict:
                query_args["where"] = filter_dict

            # Query the collection
            results = self.collection.query(**query_args)

            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                distances = results['distances'][0]
                metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(documents)

                for doc, dist, metadata in zip(documents, distances, metadatas):
                    # Convert distance to similarity score (cosine distance -> cosine similarity)
                    score = 1 - dist if dist <= 2 else 0  # Clamp to [0, 1] range

                    formatted_results.append({
                        'text': doc,
                        'score': float(score),
                        'distance': float(dist),
                        'metadata': metadata or {}
                    })

            self.logger.info(f"Retrieved {len(formatted_results)} documents")
            return formatted_results

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()

            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'persist_directory': self.index_config.chroma_dir,
                'embedding_model': self.embedding_config.model_name,
                'embedding_device': self.embedding_config.device
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {
                'collection_name': self.collection_name,
                'document_count': 0,
                'error': str(e)
            }

    def delete_collection(self) -> None:
        """
        Delete the entire collection.
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            self.logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {e}")

    def peek(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Peek at sample documents in the collection.

        Args:
            limit: Number of documents to retrieve

        Returns:
            List of sample documents
        """
        try:
            results = self.collection.peek(limit=limit)

            formatted_results = []
            if results['documents']:
                for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                    formatted_results.append({
                        'text': doc[:200] + "..." if len(doc) > 200 else doc,
                        'metadata': metadata or {},
                        'id': results['ids'][i] if results['ids'] else f"doc_{i}"
                    })

            return formatted_results

        except Exception as e:
            self.logger.error(f"Failed to peek at collection: {e}")
            return []
