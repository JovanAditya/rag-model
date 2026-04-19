"""AcademicRAG Index Exporter

Comprehensive exporter for AcademicRAG system components including
ChromaDB collections, BM25 indexes, documents, and configurations.
"""

import os
import json
import gzip
import pickle
import shutil
import tarfile
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class ExportPackage:
    """Container for exported AcademicRAG system components."""

    def __init__(
        self,
        version: str,
        timestamp: datetime,
        components: List[str],
        document_count: int,
        export_path: Path,
        metadata: Dict[str, Any]
    ):
        self.version = version
        self.timestamp = timestamp
        self.components = components
        self.document_count = document_count
        self.export_path = export_path
        self.metadata = metadata
        self.archive_path = export_path

    def __repr__(self):
        return f"ExportPackage(version={self.version}, components={self.components})"


class IndexExporter:
    """
    Comprehensive exporter for AcademicRAG system components.

    Handles export of ChromaDB collections, BM25 indexes, processed documents,
    and system configurations into portable packages for deployment and backup.
    """

    def __init__(self, output_dir: str = "./exports"):
        """
        Initialize the index exporter.

        Args:
            output_dir: Directory for export files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"IndexExporter initialized with output directory: {self.output_dir}")

    def export_complete_system(
        self,
        components: List[str],
        version_suffix: Optional[str] = None
    ) -> ExportPackage:
        """
        Export complete AcademicRAG system.

        Args:
            components: List of components to export
            version_suffix: Optional version suffix

        Returns:
            ExportPackage with export information
        """
        timestamp = datetime.now()
        version_suffix = version_suffix or timestamp.strftime("%Y%m%d_%H%M%S")
        version = f"academic_rag_v{version_suffix}"

        # Create export directory
        export_dir = self.output_dir / f"academic_rag_export_{version_suffix}"
        export_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting complete system export to: {export_dir}")
        logger.info(f"Components to export: {components}")

        exported_files = []
        document_count = 0

        # Export each component
        if "chromadb" in components:
            chroma_files, chroma_docs = self._export_chromadb(export_dir)
            exported_files.extend(chroma_files)
            document_count = max(document_count, chroma_docs)

        if "bm25" in components:
            bm25_files, bm25_docs = self._export_bm25_index(export_dir)
            exported_files.extend(bm25_files)
            document_count = max(document_count, bm25_docs)

        if "documents" in components:
            doc_files, doc_count = self._export_documents(export_dir)
            exported_files.extend(doc_files)
            document_count = max(document_count, doc_count)

        if "config" in components:
            config_files = self._export_configurations(export_dir)
            exported_files.extend(config_files)

        # Create metadata
        metadata = self._create_metadata(
            version=version,
            timestamp=timestamp,
            components=components,
            document_count=document_count,
            exported_files=exported_files
        )

        # Create archive
        archive_path = self._create_compressed_package(export_dir, metadata)

        # Clean up temporary directory
        shutil.rmtree(export_dir)

        return ExportPackage(
            version=version,
            timestamp=timestamp,
            components=components,
            document_count=document_count,
            export_path=archive_path,
            metadata=metadata
        )

    def _export_chromadb(self, export_path: Path) -> Tuple[List[Path], int]:
        """Export ChromaDB collection."""
        try:
            logger.info("Exporting ChromaDB collection...")

            # Find ChromaDB files
            chroma_dir = Path("../data/chroma_db")
            if not chroma_dir.exists():
                raise FileNotFoundError("ChromaDB directory not found at ../data/chroma_db")

            # Copy entire ChromaDB directory
            export_chroma_dir = export_path / "rag_data" / "chroma_db"
            shutil.copytree(chroma_dir, export_chroma_dir)

            # Get collection info
            collection_files = list(export_chroma_dir.rglob("*"))
            document_count = self._estimate_document_count_from_chromadb(export_chroma_dir)

            logger.info(f"✅ ChromaDB exported: {len(collection_files)} files, ~{document_count} documents")
            return collection_files, document_count

        except Exception as e:
            logger.error(f"Failed to export ChromaDB: {e}")
            raise

    def _export_bm25_index(self, export_path: Path) -> Tuple[List[Path], int]:
        """Export BM25 index with metadata."""
        try:
            logger.info("Exporting BM25 index...")

            # Find BM25 cache files
            cache_dir = Path("./cache")
            if not cache_dir.exists():
                raise FileNotFoundError("Cache directory not found")

            export_cache_dir = export_path / "rag_data" / "cache"
            shutil.copytree(cache_dir, export_cache_dir)

            # Get BM25 files and document count
            bm25_files = list(export_cache_dir.rglob("*.pkl.gz"))
            document_count = self._estimate_document_count_from_bm25(export_cache_dir)

            logger.info(f"✅ BM25 index exported: {len(bm25_files)} files, ~{document_count} documents")
            return bm25_files, document_count

        except Exception as e:
            logger.error(f"Failed to export BM25 index: {e}")
            raise

    def _export_documents(self, export_path: Path) -> Tuple[List[Path], int]:
        """Export processed document chunks."""
        try:
            logger.info("Exporting processed documents...")

            # Find processed chunks file
            chunks_file = Path("./data/processed/chunks.json")
            if not chunks_file.exists():
                raise FileNotFoundError("Processed chunks file not found")

            export_data_dir = export_path / "rag_data"
            export_data_dir.mkdir(parents=True, exist_ok=True)

            export_chunks_file = export_data_dir / "chunks.json"
            shutil.copy2(chunks_file, export_chunks_file)

            # Count documents
            with open(export_chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
                document_count = len(chunks_data) if isinstance(chunks_data, list) else 0

            logger.info(f"✅ Documents exported: {document_count} chunks")
            return [export_chunks_file], document_count

        except Exception as e:
            logger.error(f"Failed to export documents: {e}")
            raise

    def _export_configurations(self, export_path: Path) -> List[Path]:
        """Export system configurations."""
        try:
            logger.info("Exporting configurations...")

            export_config_dir = export_path / "configs"
            export_config_dir.mkdir(parents=True, exist_ok=True)

            config_files = []

            # Create RAG config
            rag_config = {
                "retrieval": {
                    "pipeline_type": "advanced",
                    "max_results": 10,
                    "use_reranking": True
                },
                "llm": {
                    "model_type": "ollama",
                    "endpoint": "http://localhost:11434",
                    "model_name": "gemini-2.5-flash"
                }
            }

            rag_config_file = export_config_dir / "rag_config.json"
            with open(rag_config_file, 'w', encoding='utf-8') as f:
                json.dump(rag_config, f, indent=2)
            config_files.append(rag_config_file)

            logger.info(f"✅ Configurations exported: {len(config_files)} files")
            return config_files

        except Exception as e:
            logger.error(f"Failed to export configurations: {e}")
            raise

    def _create_metadata(
        self,
        version: str,
        timestamp: datetime,
        components: List[str],
        document_count: int,
        exported_files: List[Path]
    ) -> Dict[str, Any]:
        """Create export metadata."""
        # Calculate total export size
        total_size = sum(f.stat().st_size for f in exported_files if f.is_file())

        metadata = {
            "version": version,
            "timestamp": timestamp.isoformat(),
            "components": components,
            "document_count": document_count,
            "export_size": self._format_size(total_size),
            "export_size_bytes": total_size,
            "file_count": len(exported_files),
            "academic_rag_version": "1.0.0",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "checksum": None  # Will be calculated after archive creation
        }

        return metadata

    def _create_compressed_package(self, export_path: Path, metadata: Dict[str, Any]) -> Path:
        """Create compressed package from export."""
        try:
            logger.info("Creating compressed package...")

            # Create archive name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"academic_rag_export_{timestamp}.tar.gz"
            archive_path = self.output_dir / archive_name

            # Create tar.gz archive
            with tarfile.open(archive_path, 'w:gz') as tar:
                tar.add(export_path, arcname=export_path.name)

            # Calculate checksum
            checksum = self._calculate_file_checksum(archive_path)
            metadata["checksum"] = checksum

            # Add metadata file to archive
            metadata_file = export_path.parent / f"{export_path.name}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

            # Add metadata to archive
            with tarfile.open(archive_path, 'w:gz') as tar:
                tar.add(export_path, arcname=export_path.name)
                tar.add(metadata_file, arcname="metadata.json")

            # Clean up metadata file
            metadata_file.unlink()

            archive_size = archive_path.stat().st_size
            logger.info(f"✅ Package created: {archive_path} ({self._format_size(archive_size)})")

            return archive_path

        except Exception as e:
            logger.error(f"Failed to create compressed package: {e}")
            raise

    def _estimate_document_count_from_chromadb(self, chroma_dir: Path) -> int:
        """Estimate document count from ChromaDB files."""
        try:
            # Look for SQLite files which contain the actual data
            sqlite_files = list(chroma_dir.rglob("*.sqlite"))
            if sqlite_files:
                # Rough estimate based on file size (1KB per document average)
                total_size = sum(f.stat().st_size for f in sqlite_files)
                return max(1, total_size // 1024)
            return 0
        except:
            return 0

    def _estimate_document_count_from_bm25(self, cache_dir: Path) -> int:
        """Estimate document count from BM25 cache files."""
        try:
            # Look for BM25 cache files
            bm25_files = list(cache_dir.rglob("*bm25*.pkl.gz"))
            if bm25_files:
                # Rough estimate based on file size
                total_size = sum(f.stat().st_size for f in bm25_files)
                return max(1, total_size // 2048)  # 2KB per document average
            return 0
        except:
            return 0

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _format_size(self, size_bytes: int) -> str:
        """Format size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"


# Import sys at the end to avoid issues
import sys
