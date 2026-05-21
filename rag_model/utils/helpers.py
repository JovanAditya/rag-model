"""Helper utilities for Advanced RAG Pipeline."""

import os
import gc
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from transformers import AutoTokenizer


def validate_query(query: str, min_length: int = 3) -> bool:
    """
    Validate search query before processing.

    Args:
        query: Search query string
        min_length: Minimum number of words required

    Returns:
        True if query is valid

    Raises:
        ValueError: If query is too short or empty
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    word_count = len(query.split())
    if word_count < min_length:
        raise ValueError(
            f"Query too short (min {min_length} words): '{query}'"
        )

    return True


def truncate_context(
    documents: List[str],
    max_length: int = 15000,
    **kwargs
) -> str:
    """
    Truncate retrieved documents to fit LLM context window.

    Args:
        documents: List of document texts (ordered by relevance)
        max_length: Maximum number of characters allowed
        **kwargs: Ignored compatibility arguments

    Returns:
        Truncated context string
    """
    context = ""
    for doc in documents:
        if len(context) + len(doc) <= max_length:
            context += doc + "\n\n---\n\n"
        else:
            # Add as much of the last document as possible without cutting mid-word
            remaining = max_length - len(context)
            if remaining > 100:
                truncated_doc = doc[:remaining]
                # Try to cut at the last space
                last_space = truncated_doc.rfind(' ')
                if last_space > 0:
                    truncated_doc = truncated_doc[:last_space]
                context += truncated_doc + "...\n\n"
            break
            
    return context.strip()


def filter_empty_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove chunks with insufficient content.

    Args:
        chunks: List of document chunks

    Returns:
        Filtered list of chunks
    """
    filtered_chunks = []
    for chunk in chunks:
        text = chunk.get('text', '').strip()
        if len(text.split()) >= 10:  # Minimum 10 words
            filtered_chunks.append(chunk)

    return filtered_chunks


def ensure_directory_exists(path: str) -> None:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def format_retrieval_results(
    results: List[Dict[str, Any]],
    add_rank: bool = True
) -> List[Dict[str, Any]]:
    """
    Format retrieval results with consistent structure.

    Args:
        results: Raw retrieval results
        add_rank: Whether to add rank information

    Returns:
        Formatted results
    """
    formatted = []
    for i, result in enumerate(results):
        formatted_result = {
            'text': result.get('text', ''),
            'score': float(result.get('score', 0.0)),
            'metadata': result.get('metadata', {}),
            'source': result.get('source', 'unknown')
        }

        if add_rank:
            formatted_result['rank'] = i + 1

        formatted.append(formatted_result)

    return formatted


def safe_gpu_memory_cleanup() -> None:
    """Safely cleanup GPU memory to avoid OOM errors."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Safely load JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded JSON data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load JSON file {file_path}: {e}")


def save_json_file(data: Any, file_path: str) -> None:
    """
    Safely save data to JSON file.

    Args:
        data: Data to save
        file_path: Output file path
    """
    ensure_directory_exists(os.path.dirname(file_path))

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise ValueError(f"Failed to save JSON file {file_path}: {e}")
