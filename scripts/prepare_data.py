#!/usr/bin/env python3
"""
Data Preparation Script for Academic RAG

This script processes PDF documents and creates:
- Chunked documents for vector store
- Metadata for document tracking
- Processed data ready for indexing

Configuration via .env file or command line arguments.

Usage:
    python scripts/prepare_data.py  # Uses .env defaults
    python scripts/prepare_data.py --input ../data/documents --output ../data/processed
    python scripts/prepare_data.py --chunk-size 1000 --chunk-overlap 200 --device cuda
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

# Load .env file
try:
    from dotenv import load_dotenv
    # Load from academic-rag/.env
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, use system env vars

# Add rag_model to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import torch
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer
    import pdfplumber
    from PIL import Image
    import io
    import base64
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("💡 Please install required packages:")
    print("   pip install torch transformers sentence-transformers pdfplumber pillow")
    sys.exit(1)

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreparator:
    """Prepare and chunk academic documents for RAG system."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, device: str = "auto"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Auto-detect device if requested
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"🔧 Initializing models (device: {self.device})")
        if self.device == "cuda":
            logger.info(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        self.tokenizer = None
        self.sentence_model = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize tokenizer and sentence transformer models."""
        try:
            # Load configuration
            import json
            # Config is in ../config/ from scripts directory
            config_path = Path(__file__).parent.parent / "config" / "data_processing.json"

            # Fallback to default config if file not found
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"📋 Loaded configuration from {config_path}")
            else:
                logger.warning(f"⚠️ Configuration file not found at {config_path}, using defaults")
                config = {
                    "embedding": {
                        "model_name": "indobenchmark/indobert-base-p2",
                        "batch_size": 32
                    }
                }

            # Initialize IndoBERT tokenizer
            model_name = config.get("embedding", {}).get("model_name", "indobenchmark/indobert-base-p2")
            logger.info(f"📥 Loading tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Initialize sentence transformer
            logger.info(f"📥 Loading sentence transformer: {model_name}")
            self.sentence_model = SentenceTransformer(model_name, device=self.device)
            batch_size = config.get("embedding", {}).get("batch_size", 32)
            logger.info(f"📊 Using batch size: {batch_size}")
            self.batch_size = batch_size

            logger.info("✅ Models initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize models: {e}")
            raise

    def _extract_images_from_page(self, page, page_num: int) -> List[Dict[str, Any]]:
        """Extract images from a PDF page using Pillow."""
        images = []
        try:
            # Try to get images from the page
            if hasattr(page, 'images'):
                for img_index, img in enumerate(page.images):
                    try:
                        # Get image bounding box
                        bbox = (img['x0'], img['top'], img['x1'], img['bottom'])

                        # Crop the page to get the image area directly from Page object
                        cropped_page = page.crop(bbox)
                        page_image = cropped_page.to_image(resolution=150)

                        # Convert to PIL Image
                        pil_image = page_image.original

                        # Get image info
                        image_info = {
                            "page": page_num + 1,
                            "image_index": img_index,
                            "width": pil_image.width,
                            "height": pil_image.height,
                            "format": pil_image.format,
                            "mode": pil_image.mode,
                            "bbox": bbox
                        }

                        # Save image to bytes
                        img_buffer = io.BytesIO()
                        pil_image.save(img_buffer, format='PNG')
                        img_buffer.seek(0)

                        # Convert to base64 for storage
                        image_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
                        image_info["image_data"] = image_base64

                        images.append(image_info)
                        logger.debug(f"   🖼️  Extracted image {img_index+1} from page {page_num+1}: {pil_image.width}x{pil_image.height}")

                    except Exception as img_error:
                        logger.debug(f"   ⚠️  Could not extract image {img_index+1} from page {page_num+1}: {img_error}")
                        continue

        except Exception as e:
            logger.debug(f"   ⚠️  Image extraction failed for page {page_num+1}: {e}")

        return images

    def _extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text and metadata from PDF file."""
        try:
            logger.info(f"📖 Processing: {pdf_path.name}")

            # Extract text using pdfplumber (better for academic documents)
            text_content = []
            extracted_images = []
            metadata = {
                "source": pdf_path.name,
                "file_path": str(pdf_path),
                "file_size": pdf_path.stat().st_size,
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            with pdfplumber.open(pdf_path) as pdf:
                pages = pdf.pages
                text_content = []

                print(f"\nEKSTRAKSI TEKS - {pdf_path.name}")
                print(f"Ukuran file: {pdf_path.stat().st_size / 1024:.1f} KB")
                print(f"Total halaman: {len(pages)}\n")

                for i, page in enumerate(pages):
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append({
                            "page": i + 1,
                            "text": page_text,
                            "bbox": page.bbox
                        })
                        print(f"\n--- Teks Halaman {i+1} ({len(page_text)} karakter) ---")
                        print(page_text.strip())
                        print("-" * 50)
                    else:
                        print(f"\n--- Teks Halaman {i+1}: 0 karakter (tidak ada teks) ---")

                    # Extract images using Pillow
                    page_images = self._extract_images_from_page(page, i)
                    extracted_images.extend(page_images)

                full_text = "\n\n".join([page["text"] for page in text_content])

                metadata.update({
                    "total_pages": len(pages),
                    "content_length": len(full_text),
                    "has_text": len(full_text.strip()) > 0,
                    "total_images": len(extracted_images),
                    "has_images": len(extracted_images) > 0
                })

                # Image detection summary
                print(f"\nDETEKSI GAMBAR - {pdf_path.name}")
                if extracted_images:
                    for idx, img in enumerate(extracted_images):
                        print(f"Gambar {idx+1} pada Halaman {img['page']}: Dimensi {img['width']}x{img['height']}, Mode {img.get('mode','?')}, Format Output PNG")
                        b64 = img.get('image_data', '')
                        if b64:
                            b64_preview = b64[:60] + "...[BASE64 TRUNCATED]..." + b64[-20:]
                            print(f"   Base64 Preview: {b64_preview} (Panjang total: {len(b64):,} karakter)")
                else:
                    print("Tidak ada gambar terdeteksi.")

                print(f"\nPENGGABUNGAN TEKS - {pdf_path.name}")
                word_count = len(full_text.split())
                print(f"Total karakter: {len(full_text):,}")
                print(f"Estimasi kata: {word_count:,}")
                print(f"Halaman dengan teks: {len(text_content)} / {len(pages)}")
                if full_text.strip():
                    print(f"\n--- Teks Gabungan Keseluruhan ---")
                    print(full_text.strip())
                    print("-" * 50)

            if len(full_text.strip()) == 0 and not extracted_images:
                logger.warning(f"⚠️  No text or images extracted from {pdf_path.name}")
                return {"error": "No content extracted", "metadata": metadata}

            return {
                "text": full_text,
                "metadata": metadata,
                "pages": len(pages),
                "images": extracted_images
            }

        except Exception as e:
            logger.error(f"❌ Error processing {pdf_path.name}: {e}")
            return {"error": str(e), "metadata": {"source": pdf_path.name}}

    def _chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into chunks using tokenizer."""
        try:
            # Tokenize text
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            token_count = len(tokens)

            chunks = []

            if token_count <= self.chunk_size:
                # Text is shorter than chunk size, return as single chunk
                chunk_text = self.tokenizer.decode(tokens)
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "token_count": token_count,
                        "char_count": len(chunk_text)
                    }
                })
            else:
                # Split into overlapping chunks
                step_size = self.chunk_size - self.chunk_overlap
                total_chunks = (token_count - self.chunk_overlap) // step_size + 1

                for i in range(total_chunks):
                    start_token = i * step_size
                    end_token = min(start_token + self.chunk_size, token_count)

                    chunk_tokens = tokens[start_token:end_token]
                    chunk_text = self.tokenizer.decode(chunk_tokens)

                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            **metadata,
                            "chunk_index": i,
                            "total_chunks": total_chunks,
                            "token_start": start_token,
                            "token_end": end_token,
                            "token_count": len(chunk_tokens),
                            "char_count": len(chunk_text),
                            "overlap_tokens": self.chunk_overlap if i > 0 else 0
                        }
                    })

            return chunks

        except Exception as e:
            logger.error(f"❌ Error chunking text: {e}")
            return []

    def process_documents(self, input_dir: str, output_dir: str) -> bool:
        """Process all PDF documents and create chunks."""
        logger.info(f"📁 Input directory: {input_dir}")
        logger.info(f"📁 Output directory: {output_dir}")

        input_path = Path(input_dir)
        output_path = Path(output_dir)

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all PDF files
        pdf_files = list(input_path.glob("*.pdf"))

        if not pdf_files:
            logger.error("❌ No PDF files found in input directory")
            return False

        logger.info(f"📄 Found {len(pdf_files)} PDF files")

        # Process all documents
        all_chunks = []
        all_metadata = []
        failed_files = []

        start_time = time.time()

        for pdf_path in pdf_files:
            # Extract text and metadata
            result = self._extract_text_from_pdf(pdf_path)

            if "error" in result:
                failed_files.append({
                    "file": pdf_path.name,
                    "error": result["error"]
                })
                continue

            text = result["text"]
            metadata = result["metadata"]

            # Skip very short documents
            if len(text.strip()) < 100:
                logger.warning(f"⚠️  Skipping very short document: {pdf_path.name}")
                metadata["processing_status"] = "skipped_too_short"
                all_metadata.append(metadata)
                continue

            # Chunk the text
            chunks = self._chunk_text(text, metadata)

            if not chunks:
                failed_files.append({
                    "file": pdf_path.name,
                    "error": "Failed to chunk text"
                })
                metadata["processing_status"] = "chunking_failed"
                all_metadata.append(metadata)
                continue

            # Display chunking results
            print(f"\nCHUNKING - {pdf_path.name}")
            print(f"Chunk size: {self.chunk_size} token")
            print(f"Overlap: {self.chunk_overlap} token")
            print(f"Total chunk: {len(chunks)}\n")

            for idx, chunk in enumerate(chunks):
                chunk_id = chunk.get('metadata', {}).get('chunk_index', idx)
                token_count = chunk['metadata'].get('token_count', '?')
                char_count = len(chunk['text'])
                print(f"\n--- Isi Teks Chunk {chunk_id} ({token_count} token, {char_count} karakter) ---")
                print(chunk['text'].strip())
                print("-" * 50)

            if len(chunks) > 1:
                print(f"\nDemonstrasi Overlap (Chunk 0 -> Chunk 1):")
                c0_tokens = self.tokenizer.encode(chunks[0]['text'], add_special_tokens=False)
                c0_overlap_text = self.tokenizer.decode(c0_tokens[-self.chunk_overlap:]).strip()
                print(f"Chunk 0:\n... {c0_overlap_text}")
                print("-" * 50)
                
                c1_tokens = self.tokenizer.encode(chunks[1]['text'], add_special_tokens=False)
                c1_overlap_text = self.tokenizer.decode(c1_tokens[:self.chunk_overlap]).strip()
                print(f"Chunk 1:\n{c1_overlap_text} ...")
                print("-" * 50)

            # Add chunks to collection
            all_chunks.extend(chunks)

            # Update document metadata
            metadata.update({
                "processing_status": "success",
                "chunks_created": len(chunks),
                "total_tokens": sum(chunk["metadata"]["token_count"] for chunk in chunks)
            })
            all_metadata.append(metadata)

        processing_time = time.time() - start_time

        # Create chunks.json
        chunks_file = output_path / "chunks.json"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        # Create metadata.json
        metadata_file = output_path / "metadata.json"
        processing_metadata = {
            "total_documents": len(pdf_files),
            "processed_documents": len([m for m in all_metadata if m.get("processing_status") == "success"]),
            "failed_documents": len(failed_files),
            "total_chunks": len(all_chunks),
            "total_tokens": sum(m.get("total_tokens", 0) for m in all_metadata),
            "processing_time": processing_time,
            "chunk_parameters": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "device": self.device
            },
            "model_info": {
                "tokenizer": "indobenchmark/indobert-base-p2",
                "sentence_transformer": "indobenchmark/indobert-base-p2"
            },
            "failed_files": failed_files,
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(processing_metadata, f, ensure_ascii=False, indent=2)

        # Print summary
        logger.info("📊 Processing Summary:")
        logger.info(f"   Total documents: {len(pdf_files)}")
        logger.info(f"   Processed successfully: {processing_metadata['processed_documents']}")
        logger.info(f"   Failed: {processing_metadata['failed_documents']}")
        logger.info(f"   Total chunks: {processing_metadata['total_chunks']}")
        logger.info(f"   Total tokens: {processing_metadata['total_tokens']}")
        logger.info(f"   Processing time: {processing_time:.2f}s")
        logger.info(f"   Output files: {chunks_file.name}, {metadata_file.name}")

        # Return success status
        success = processing_metadata['failed_documents'] == 0
        if success:
            logger.info("✅ All documents processed successfully!")
        else:
            logger.warning(f"⚠️  {processing_metadata['failed_documents']} documents failed")

        return success

    def generate_embeddings(self, chunks_file: str, output_dir: str) -> bool:
        """Generate embeddings for all chunks."""
        try:
            logger.info("🔢 Generating embeddings...")

            chunks_path = Path(chunks_file)
            embeddings_file = Path(output_dir) / "embeddings.json"

            # Load chunks
            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)

            # Generate embeddings in batches
            batch_size = 32
            all_embeddings = []
            all_texts = [chunk["text"] for chunk in chunks]

            logger.info(f"📝 Processing {len(chunks)} chunks...")

            for i in range(0, len(all_texts), batch_size):
                batch_texts = all_texts[i:i+batch_size]

                with torch.no_grad():
                    batch_embeddings = self.sentence_model.encode(
                        batch_texts,
                        convert_to_tensor=True,
                        show_progress_bar=True
                    )

                    if batch_embeddings.device != torch.device('cpu'):
                        batch_embeddings = batch_embeddings.cpu()

                    all_embeddings.extend(batch_embeddings.numpy().tolist())

                logger.info(f"   Processed chunk {i+1}-{min(i+batch_size, len(all_texts))}")

            # Save embeddings
            embeddings_data = {
                "embeddings": all_embeddings,
                "texts": all_texts,
                "model": "indobenchmark/indobert-base-p2",
                "dimension": len(all_embeddings[0]) if all_embeddings else 0,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(embeddings_file, 'w', encoding='utf-8') as f:
                json.dump(embeddings_data, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ Embeddings saved to {embeddings_file}")
            logger.info(f"   Embedding dimension: {embeddings_data['dimension']}")

            return True

        except Exception as e:
            logger.error(f"❌ Error generating embeddings: {e}")
            return False


def main():
    """Main function."""
    # Get defaults from environment variables
    default_input = os.getenv("DOCUMENTS_DIR", "../data/documents")
    default_output = os.getenv("PROCESSED_DIR", "../data/processed")
    default_chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    default_chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    default_device = os.getenv("EMBEDDING_DEVICE", "auto")

    parser = argparse.ArgumentParser(
        description="Prepare academic documents for RAG system"
    )

    parser.add_argument(
        "--input", "-i",
        default=default_input,
        help=f"Input directory containing PDF documents (default: {default_input})"
    )

    parser.add_argument(
        "--output", "-o",
        default=default_output,
        help=f"Output directory for processed data (default: {default_output})"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=default_chunk_size,
        help=f"Maximum tokens per chunk (default: {default_chunk_size})"
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=default_chunk_overlap,
        help=f"Token overlap between chunks (default: {default_chunk_overlap})"
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps", "auto"],
        default=default_device,
        help=f"Device for processing (default: {default_device})"
    )

    parser.add_argument(
        "--generate-embeddings",
        action="store_true",
        help="Generate pre-computed embeddings"
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip processing if output files already exist"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocess all files (overwrite existing)"
    )

    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up old processed data after successful processing (use with caution)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("🎓 ACADEMIC RAG DATA PREPARATION")
    print("=" * 50)
    print(f"📁 Input: {args.input}")
    print(f"📁 Output: {args.output}")
    print(f"📏 Chunk size: {args.chunk_size} tokens")
    print(f"🔗 Chunk overlap: {args.chunk_overlap} tokens")
    print(f"💻 Device: {args.device}")
    print()

    # Validate input directory
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"❌ Input directory does not exist: {input_path}")
        sys.exit(1)

    if not input_path.is_dir():
        logger.error(f"❌ Input path is not a directory: {input_path}")
        sys.exit(1)

    # Check for PDF files
    pdf_files = list(input_path.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"❌ No PDF files found in: {input_path}")
        sys.exit(1)

    print(f"📄 Found {len(pdf_files)} PDF files to process")
    print()

    # Check if output files already exist (for --skip-existing)
    output_path = Path(args.output)
    chunks_file = output_path / "chunks.json"
    metadata_file = output_path / "metadata.json"

    if args.skip_existing and chunks_file.exists() and metadata_file.exists():
        print("✅ Output files already exist, skipping processing (use --force to override)")
        print(f"📁 Chunks: {chunks_file}")
        print(f"📁 Metadata: {metadata_file}")
        print()
        print("💡 To force reprocessing, use: --force")
        print("💡 To continue with existing data, proceed to index building")
        sys.exit(0)

    # Initialize data preparator
    try:
        preparator = DataPreparator(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            device=args.device
        )
    except Exception as e:
        logger.error(f"❌ Failed to initialize data preparator: {e}")
        sys.exit(1)

    # Process documents
    start_time = time.time()

    try:
        # If --force, remove existing output files
        if args.force:
            if chunks_file.exists():
                print(f"🗑️  Removing existing chunks file: {chunks_file}")
                chunks_file.unlink()
            if metadata_file.exists():
                print(f"🗑️  Removing existing metadata file: {metadata_file}")
                metadata_file.unlink()

        success = preparator.process_documents(args.input, args.output)

        if success and args.generate_embeddings:
            # Generate embeddings if requested
            chunks_file = Path(args.output) / "chunks.json"
            if chunks_file.exists():
                preparator.generate_embeddings(str(chunks_file), args.output)

        processing_time = time.time() - start_time

        print("\n" + "=" * 50)
        print("📊 FINAL SUMMARY")
        print("=" * 50)

        if success:
            print("✅ Data preparation completed successfully!")
            print(f"⏱️  Total time: {processing_time:.2f}s")
            print()
            print("💡 Next steps:")
            print("   1. Build indexes: python scripts/build_indexes.py")
            print("   2. Verify system: python scripts/verify_system.py")
            print("   3. Test usage: python examples/basic_usage.py")
        else:
            print("❌ Data preparation completed with errors!")
            print(f"⏱️  Total time: {processing_time:.2f}s")
            print()
            print("💡 Please check the error messages above and:")
            print("   1. Verify PDF files are readable")
            print("   2. Check available disk space")
            print("   3. Ensure all dependencies are installed")

        print("=" * 50)

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("\n⚠️  Data preparation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
