"""
Pytest configuration for Advanced RAG Pipeline testing.

This file contains shared fixtures and configuration for all tests.
"""

import os
import sys
import tempfile
import pytest
import json
from unittest.mock import Mock, patch

# Add src to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock external services for testing
pytest_plugins = [
    "pytest.mock",
    "pytest.fixture",
    "pytest.markers",
]

# Markers for test categorization
pytest.mark.unit = pytest.mark.unit  # Unit tests
pytest.mark.integration = pytest.mark.integration  # Integration tests
pytest.mark.e2e = pytest.mark.e2e  # End-to-end tests
pytest.mark.slow = pytest.mark.slow  # Slow tests
pytest.mark.gpu = pytest.mark.gpu  # Tests requiring GPU

# Skip GPU tests unless GPU is available
def pytest_runtest_setup(item):
    """Skip GPU tests if GPU is not available."""
    if item.get_closest_marker("gpu"):
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("GPU not available - skipping GPU test")
        except ImportError:
            pytest.skip("PyTorch not available - skipping GPU test")

pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment variables."""
    osmodel/.environ.setdefault('EMBEDDING_DEVICE', 'cpu')
    osmodel/.environ.setdefault('TESTING', 'true')
    osmodel/.environ.setdefault('LOG_LEVEL', 'DEBUG')
    osmodel/.environ.setdefault('RANDOM_SEED', '42')

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_path:
        yield temp_path

@pytest.fixture
def sample_chunks():
    """Sample document chunks for testing."""
    return [
        {
            "text": "This is the first chunk about university registration procedures. Students can register online through the student portal.",
            "metadata": {
                "source": "test_registration.pdf",
                "chunk_id": "chunk_1",
                "page": 1,
                "start_char": 0,
                "end_char": 150
            }
        },
        {
            "text": "This is the second chunk describing academic calendar and important dates for the current semester.",
            "metadata": {
                "source": "test_calendar.pdf",
                "chunk_id": "chunk_1",
                "page": 1,
                "start_char": 0,
                "end_char": 130
            }
        },
        {
            "text": "This is the third chunk about tuition fees and payment options available to students.",
            "metadata": {
                "source": "test_tuition.pdf",
                "chunk_id": "chunk_1",
                "page": 1,
                "start_char": 0,
                "end_char": 110
            }
        }
    ]

@pytest.fixture
def sample_queries():
    """Sample evaluation queries for testing."""
    return [
        {
            "query_id": "Q001",
            "query": "Bagaimana cara mendaftar mata kuliah baru?",
            "relevant_docs": ["test_registration.pdf"],
            "ground_truth": ["Pendaftaran dilakukan secara online melalui sistem informasi akademik dengan menggunakan NIM dan password."]
        },
        {
            "query_id": "Q002",
            "query": "Berapa saja biaya kuliah per semester?",
            "relevant_docs": ["test_tuition.pdf"],
            "ground_truth": ["Biaya kuliah bervariasi tergantung program studi dan jumlah SKS yang diambil."]
        },
        {
            "query_id": "Q003",
            "query": "Kapan jadwal kuliah semester ganjil dimulai?",
            "relevant_docs": ["test_calendar.pdf"],
            "ground_truth": ["Semester ganjil biasanya dimulai pada bulan Agustus dengan orientation mahasiswa baru."]
        }
    ]

@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    mock_embedding_config = Mock()
    mock_embedding_config.model_name = "sentence-transformers/all-MiniLM-L6-v2"
    mock_embedding_config.device = "cpu"
    mock_embedding_config.batch_size = 16

    mock_llm_config = Mock()
    mock_llm_config.model_type = "qwen"
    mock_llm_config.temperature = 0.1
    mock_llm_config.max_tokens = 100

    mock_es_config = Mock()
    mock_es_config.host = "localhost"
    mock_es_config.port = 9200
    mock_es_config.index_name = "test_index"

    mock_chroma_config = Mock()
    mock_chroma_config.persist_directory = "/tmp/test_chroma"
    mock_chroma_config.collection_name = "test_collection"

    return {
        "embedding": mock_embedding_config,
        "llm": mock_llm_config,
        "elasticsearch": mock_es_config,
        "chroma": mock_chroma_config
    }

@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger

@pytest.fixture
def mock_data_directory(temp_dir):
    """Create mock data directory structure."""
    data_dir = os.path.join(temp_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    for subdir in ["raw", "processed", "evaluation"]:
        os.makedirs(os.path.join(data_dir, subdir), exist_ok=True)

    return data_dir

# Mock external services
@pytest.fixture(autouse=True)
@patch('src.indexing.elasticsearch_index.ElasticsearchIndex')
def mock_elasticsearch_index():
    """Mock ElasticsearchIndex for testing."""
    mock_es = Mock()
    mock_es.health_check.return_value = {"healthy": True}
    mock_es.create_index.return_value = True
    mock_es.index_documents.return_value = True
    mock_es.search.return_value = []
    mock_es.get_index_stats.return_value = {"document_count": 0}
    return mock_es

@pytest.fixture(autouse=True)
@patch('src.indexing.vector_store.VectorStore')
def mock_vector_store():
    """Mock VectorStore for testing."""
    mock_vs = Mock()
    mock_vs.add_documents.return_value = True
    mock_vs.similarity_search.return_value = []
    mock_vs.get_collection_stats.return_value = {"document_count": 0}
    return mock_vs

@pytest.fixture
def mock_model():
    """Mock embedding model for testing."""
    mock_model = Mock()
    mock_model.encode.return_value = [[0.1, 0.2, 0.3]]  # Mock embedding
    mock_model.get_sentence_embedding_dimension.return_value = 384
    return mock_model

@pytest.fixture
def mock_cross_encoder():
    """Mock cross-encoder reranker for testing."""
    mock_reranker = Mock()
    mock_reranker.predict.return_value = [[0.8, 0.2], [0.6, 0.4]]
    return mock_reranker

@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    mock_llm = Mock()
    mock_llm.generate.return_value = {
        "answer": "This is a test response.",
        "model": "qwen3:8b",
        "tokens_used": 50,
        "generation_time": 2.0
    }
    return mock_llm

# Configure test discovery
collect_ignore = [
    "setup.py",
    "conftest.py",
    "__pycache__",
    ".*pytest_cache*",
    "htmlcov",
    ".coverage*",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    "venv",
    "env",
    "site-packages",
    "node_modules"
]

# Configure test output
def pytest_configure(config):
    """Configure pytest settings."""
    config.addinivalue_line("python_files = test_*.py")
    config.addinivalue_line("python_classes = Test*")
    config.addinivalue_line("python_functions = test_*")
    config.addinivalue_line("testpaths = tests")
    config.addinvalue_line("addopts = --strict-markers")

    # Markers configuration
    config.addinivalue_line("markers = unit: Unit tests")
    config.addinivalue_line("markers = integration: Integration tests")
    config.addinivalue_line("markers = e2e: End-to-end tests")
    config.addinvalue_line("markers = slow: Slow tests")
    config.addinvalue_line("markers = gpu: Tests requiring GPU")

# Custom markers for test filtering
def pytest_collection_modifyitems(items, config):
    """Add custom markers and modify test collection."""
    for item in items:
        # Add slow marker to tests that might take > 5 seconds
        item.add_marker(pytest.mark.slow if "benchmark" in item.nodeid else None)

        # Add GPU marker to GPU-dependent tests
        if "gpu" in item.keywords or "cuda" in item.nodeid:
            item.add_marker(pytest.mark.gpu)

        # Add integration marker for tests that use external services
        if "integration" in item.nodeid or "service" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Add unit marker for isolated component tests
        if item.get_closest_marker(None) is None:  # No existing markers
            item.add_marker(pytest.mark.unit)

# Custom helpers for testing
class TestDataGenerator:
    """Helper class for generating test data."""

    @staticmethod
    def create_chunks(count=5, text_length=100):
        """Create test document chunks."""
        chunks = []
        for i in range(count):
            chunks.append({
                "text": f"Test chunk {i+1} with approximately {text_length} characters. " * (text_length // 20),
                "metadata": {
                    "source": f"test_doc_{(i//3)+1}.pdf",
                    "chunk_id": f"chunk_{i+1}",
                    "page": (i % 3) + 1,
                    "start_char": i * 200,
                    "end_char": (i + 1) * 200
                }
            })
        return chunks

    @staticmethod
    def create_queries(count=5, topics=None):
        """Create test evaluation queries."""
        if topics is None:
            topics = ["registration", "tuition", "calendar", "graduation", "evaluation"]

        queries = []
        for i in range(count):
            topic = topics[i % len(topics)]
            queries.append({
                "query_id": f"TEST_{i+1:03d}",
                "query": f"Test question about {topic}",
                "relevant_docs": [f"doc_{(i%3)+1}.pdf"],
                "ground_truth": [f"Test ground truth for {topic} question."]
            })
        return queries

    @staticmethod
    def save_json(data, file_path):
        """Save data to JSON file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_json(file_path):
        """Load data from JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)

class MockResponse:
    """Helper class for creating mock HTTP responses."""

    @staticmethod
    def json_response(data, status_code=200):
        """Create a mock JSON response."""
        from unittest.mock import Mock
        response = Mock()
        response.status_code = status_code
        response.json.return_value = data
        return response

    @staticmethod
    def text_response(text, status_code=200):
        """Create a mock text response."""
        from unittest.mock import Mock
        response = Mock()
        response.status_code = status_code
        response.text = text
        response.json.return_value = {"text": text}
        return response

class TestEnvironment:
    """Helper class for managing test environment setup."""

    @staticmethod
    def setup_test_data_structure(base_dir):
        """Create standard test data directory structure."""
        directories = [
            "data/raw",
            "data/processed",
            "data/evaluation",
            "results",
            "logs",
            "cache"
        ]

        for directory in directories:
            os.makedirs(os.path.join(base_dir, directory), exist_ok=True)

    @staticmethod
    def create_sample_documents(base_dir):
        """Create sample PDF documents for testing."""
        documents = [
            ("registration_guide.pdf", "University registration procedures and guidelines."),
            ("academic_calendar.pdf", "Important dates and academic calendar."),
            ("tuition_fees.pdf", "Tuition fees and payment information."),
            ("student_handbook.pdf", "Student handbook with policies.")
        ]

        raw_dir = os.path.join(base_dir, "data", "raw")
        for filename, content in documents:
            file_path = os.path.join(raw_dir, filename)
            with open(file_path, 'w') as f:
                f.write(content)

    @staticmethod
    def create_evaluation_dataset(base_dir, num_queries=5):
        """Create sample evaluation dataset."""
        queries = TestDataGenerator.create_queries(num_queries)
        eval_dir = os.path.join(base_dir, "data", "evaluation")

        with open(os.path.join(eval_dir, "queries.json"), 'w') as f:
            json.dump(queries, f, indent=2)

# Global test utilities
def assert_valid_chunk_structure(chunk):
    """Assert that chunk has required structure."""
    assert isinstance(chunk, dict)
    assert "text" in chunk
    assert "metadata" in chunk
    assert isinstance(chunk["text"], str)
    assert isinstance(chunk["metadata"], dict)

    required_metadata = ["source", "chunk_id", "page"]
    for field in required_metadata:
        assert field in chunk["metadata"]
    assert chunk["metadata"][field] is not None

def assert_valid_query_structure(query):
    """Assert that query has required structure."""
    assert isinstance(query, dict)
    required_fields = ["query_id", "query", "relevant_docs"]
    for field in required_fields:
        assert field in query
        assert query[field] is not None

def assert_valid_result_structure(result):
    """Assert that pipeline result has required structure."""
    assert isinstance(result, dict)
    required_fields = ["query_id", "query", "retrieved_docs", "ground_truth"]
    for field in required_fields:
        assert field in result

    assert isinstance(result["retrieved_docs"], list)
    for doc in result["retrieved_docs"]:
        assert_valid_chunk_structure(doc)

def assert_valid_performance_metrics(metrics):
    """Assert that performance metrics have valid structure."""
    assert isinstance(metrics, dict)

    if "mrr" in metrics:
        assert isinstance(metrics["mrr"], (int, float))
        assert 0 <= metrics["mrr"] <= 1.0

    if "precision_at_k" in metrics:
        assert isinstance(metrics["precision_at_k"], dict)
        for k, score in metrics["precision_at_k"].items():
            assert isinstance(k, int)
            assert isinstance(score, (int, float))
            assert 0 <= score <= 1.0

    if "recall_at_k" in metrics:
        assert isinstance(metrics["recall_at_k"], dict)
        for k, score in metrics["recall_at_k"].items():
            assert isinstance(k, int)
            assert isinstance(score, (int, float))
            assert 0 <= score <= 1.0