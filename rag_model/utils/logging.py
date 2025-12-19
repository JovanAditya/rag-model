"""Logging utilities for Advanced RAG Pipeline."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from rich.logging import RichHandler
from rich.console import Console


class PipelineLogger:
    """
    Centralized logging for RAG pipeline with rich formatting.
    """

    def __init__(
        self,
        name: str = "rag_pipeline",
        log_file: str = "pipeline.log",
        level: str = "INFO",
        enable_rich: bool = True
    ):
        """
        Initialize pipeline logger.

        Args:
            name: Logger name
            log_file: Path to log file
            level: Logging level
            enable_rich: Enable rich console output
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Clear existing handlers
        self.logger.handlers.clear()

        # Console handler
        if enable_rich:
            console = Console(stderr=True)
            console_handler = RichHandler(
                console=console,
                show_time=True,
                show_path=False,
                markup=True,
                rich_tracebacks=True
            )
        else:
            console_handler = logging.StreamHandler(sys.stderr)

        console_handler.setLevel(getattr(logging, level.upper()))

        # File handler
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # File gets all logs

        # Formatter
        detailed_formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(detailed_formatter)

        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        self.logger.info(f"Logger initialized: {name}")
        self.logger.info(f"Log file: {log_path.absolute()}")

    def log_retrieval(
        self,
        query_id: str,
        query: str,
        num_retrieved: int,
        latency: float,
        component: str = "Retriever"
    ) -> None:
        """Log retrieval operation."""
        self.logger.info(
            f"[{component}] Query {query_id}: \"{query[:50]}...\" | "
            f"Retrieved: {num_retrieved} docs | Latency: {latency:.3f}s"
        )

    def log_generation(
        self,
        query_id: str,
        tokens_used: int,
        latency: float,
        model: str,
        component: str = "LLMGenerator"
    ) -> None:
        """Log LLM generation."""
        self.logger.info(
            f"[{component}] Query {query_id} | Model: {model} | "
            f"Tokens: {tokens_used} | Latency: {latency:.3f}s"
        )

    def log_error(
        self,
        component: str,
        error: Exception,
        context: Optional[Dict] = None
    ) -> None:
        """Log error with context."""
        error_msg = f"[{component}] {type(error).__name__}: {str(error)}"
        if context:
            error_msg += f" | Context: {context}"
        self.logger.error(error_msg, exc_info=True)

    def error(self, message: str, *args, **kwargs) -> None:
        """Direct error method for compatibility with standard logger interface."""
        self.logger.error(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        """Direct info method for compatibility with standard logger interface."""
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        """Direct warning method for compatibility with standard logger interface."""
        self.logger.warning(message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs) -> None:
        """Direct debug method for compatibility with standard logger interface."""
        self.logger.debug(message, *args, **kwargs)

    def log_evaluation_start(self, dataset_size: int, pipeline: str) -> None:
        """Log evaluation start."""
        self.logger.info(f"[Evaluator] Starting evaluation for {pipeline} pipeline")
        self.logger.info(f"[Evaluator] Dataset size: {dataset_size} queries")

    def log_evaluation_progress(self, current: int, total: int, metric: str) -> None:
        """Log evaluation progress."""
        progress = (current / total) * 100
        self.logger.info(f"[Evaluator] Progress: {current}/{total} ({progress:.1f}%) | {metric}")

    def log_evaluation_summary(self, results: Dict[str, float]) -> None:
        """Log evaluation summary."""
        self.logger.info("[Evaluator] Evaluation completed!")
        for metric, score in results.items():
            self.logger.info(f"[Evaluator] {metric}: {score:.4f}")


def setup_logger(
    name: str = "rag_pipeline",
    log_file: str = "pipeline.log",
    level: str = "INFO",
    enable_rich: bool = True
) -> PipelineLogger:
    """
    Setup and return a pipeline logger.

    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        enable_rich: Enable rich console output

    Returns:
        Configured PipelineLogger instance
    """
    return PipelineLogger(name, log_file, level, enable_rich)