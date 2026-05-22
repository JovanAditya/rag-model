import os
import json
from pathlib import Path

# Load .env from the academic-rag folder
try:
    from dotenv import load_dotenv
    # Find .env relative to this config.py file's location
    config_dir = Path(__file__).parent  # rag_model/core/
    rag_model_dir = config_dir.parent     # rag_model/
    academic_rag_dir = rag_model_dir.parent  # academic-rag/
    env_path = academic_rag_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Fallback: try current directory
        load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Run: pip install python-dotenv")
try:
    import yaml
except ImportError:
    yaml = None
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Literal
from pathlib import Path

# Type definitions
LLMType = Literal["gemini", "ollama"]
PipelineType = Literal["baseline", "advanced"]
DeviceType = Literal["cpu", "cuda"]


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    model_name: str = os.getenv("EMBEDDING_MODEL", "indobenchmark/indobert-base-p2")
    device: DeviceType = os.getenv("EMBEDDING_DEVICE", "cpu")
    batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    max_seq_length: int = 512
    cache_dir: Optional[str] = None


@dataclass
class LLMConfig:
    """Configuration for Language Models."""
    model_type: str = os.getenv("LLM_PROVIDER", "gemini")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4000"))
    top_p: float = 0.9
    _api_key: Optional[str] = None
    endpoint: Optional[str] = None
    ollama_endpoint: Optional[str] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def get_model_name(self) -> str:
        """Get full model name based on type and .env configuration."""
        # Use environment variable if available, otherwise use defaults
        env_model = os.getenv(f"{self.model_type.upper()}_MODEL")
        if env_model:
            return env_model

        model_names = {
            "gemini": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            "ollama": os.getenv("OLLAMA_MODEL", "llama3.2:latest"),
        }
        return model_names.get(self.model_type, "gemini-2.5-flash")

    def get_api_key(self) -> Optional[str]:
        """Get API key for the current LLM provider."""
        if self.model_type == "gemini":
            return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return None

    @property
    def api_key(self) -> Optional[str]:
        """Get API key for the current LLM provider."""
        return self.get_api_key()

    @property
    def needs_api_key(self) -> bool:
        """Check if LLM requires API key."""
        return self.model_type in ["gemini"]


@dataclass
class RerankerConfig:
    """Configuration for cross-encoder reranking."""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_length: int = 512
    device: DeviceType = os.getenv("RERANKER_DEVICE", "cpu")
    batch_size: int = 32
    enable_cache: bool = True
    score_threshold: float = 0.0


@dataclass
class RetrievalConfig:
    """Configuration for retrieval components."""
    pipeline_type: PipelineType = os.getenv("PIPELINE_TYPE", "advanced")
    max_results: int = int(os.getenv("MAX_RESULTS", "5"))
    use_reranking: bool = os.getenv("USE_RERANKING", "true").lower() == "true"

    # Advanced pipeline settings
    bm25_k: int = 100  # Increased from 50 to improve recall
    vector_k: int = 100  # Increased from 50 to improve recall
    rerank_k: int = 40  # Increased from 20 to provide more candidates for reranking
    rrf_k: int = 60  # RRF constant for fusion
    bm25_weight: float = float(os.getenv("BM25_WEIGHT", "0.4"))
    vector_weight: float = float(os.getenv("VECTOR_WEIGHT", "0.6"))
    fusion_strategy: str = os.getenv("FUSION_STRATEGY", "rrf")


@dataclass
class BM25Config:
    """Configuration for BM25 index."""
    k1: float = float(os.getenv("BM25_K1", "1.5"))
    b: float = float(os.getenv("BM25_B", "0.75"))
    ngram_range_min: int = int(os.getenv("BM25_NGRAM_MIN", "1"))
    ngram_range_max: int = int(os.getenv("BM25_NGRAM_MAX", "2"))
    min_df: int = int(os.getenv("BM25_MIN_DF", "1"))
    max_df: float = float(os.getenv("BM25_MAX_DF", "0.95"))
    sublinear_tf: bool = os.getenv("BM25_SUBLINEAR_TF", "true").lower() == "true"


@dataclass
class IndexConfig:
    """Configuration for unified indexes (ChromaDB + BM25)."""
    chroma_dir: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "../data/chroma_db")
    chroma_collection: str = os.getenv("COLLECTION_NAME", "academic_docs")
    cache_dir: str = os.getenv("INDEX_CACHE_DIR", "../data/cache")
    enable_unified_cache: bool = True


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    metrics: List[str] = field(default_factory=lambda: ["mrr", "precision_at_k", "recall_at_k"])
    k_values: List[int] = field(default_factory=lambda: [3, 5, 10])
    use_ragas: bool = True
    ragas_judge_model: str = "gemini"


@dataclass
class RAGConfig:
    """Main configuration class for Academic RAG Model."""

    # Component configurations
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Global settings
    log_level: str = "INFO"
    cache_enabled: bool = True
    timeout_seconds: int = 30

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'RAGConfig':
        """Create configuration from dictionary."""
        # Extract nested configurations
        embedding_config = EmbeddingConfig(**config_dict.get('embedding', {}))
        llm_config = LLMConfig(**config_dict.get('llm', {}))
        retrieval_config = RetrievalConfig(**config_dict.get('retrieval', {}))
        index_config = IndexConfig(**config_dict.get('index', {}))
        bm25_config = BM25Config(**config_dict.get('bm25', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))

        # Global settings
        global_settings = {
            'log_level': config_dict.get('log_level', 'INFO'),
            'cache_enabled': config_dict.get('cache_enabled', True),
            'timeout_seconds': config_dict.get('timeout_seconds', 30)
        }

        return cls(
            embedding=embedding_config,
            llm=llm_config,
            retrieval=retrieval_config,
            index=index_config,
            bm25=bm25_config,
            evaluation=evaluation_config,
            **global_settings
        )

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'RAGConfig':
        """Load configuration from JSON or YAML file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

        return cls.from_dict(config_dict)

    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Load configuration from environment variables."""
        config_dict = {}

        # Helper function to get env variable without prefix since the .env keys don't use it consistently
        def get_env_var(key: str, default=None, var_type=str):
            value = os.getenv(key, default)
            if value is not None:
                if var_type == bool:
                    if isinstance(value, bool):
                        return value
                    return value.lower() in ('true', '1', 'yes', 'on')
                elif var_type == int:
                    return int(value)
                elif var_type == float:
                    return float(value)
                return var_type(value)
            return default

        # Embedding config
        config_dict['embedding'] = {
            'model_name': get_env_var('EMBEDDING_MODEL', 'indobenchmark/indobert-base-p2'),
            'device': get_env_var('EMBEDDING_DEVICE', 'cpu'),
            'batch_size': get_env_var('EMBEDDING_BATCH_SIZE', 32, int),
            'max_seq_length': get_env_var('EMBEDDING_MAX_SEQ_LENGTH', 512, int),
            'cache_dir': get_env_var('EMBEDDING_CACHE_DIR', None)
        }

        # LLM config
        config_dict['llm'] = {
            'model_type': get_env_var('LLM_PROVIDER', 'gemini'),
            'temperature': get_env_var('LLM_TEMPERATURE', 0.2, float),
            'max_tokens': get_env_var('LLM_MAX_TOKENS', 4000, int),
            'top_p': get_env_var('LLM_TOP_P', 0.9, float),
            '_api_key': get_env_var('GEMINI_API_KEY', None),  # For Gemini
            'endpoint': get_env_var('OLLAMA_BASE_URL', None)  # For Ollama
        }

        # Retrieval config
        config_dict['retrieval'] = {
            'pipeline_type': get_env_var('PIPELINE_TYPE', 'advanced'),
            'max_results': get_env_var('MAX_RESULTS', 5, int),
            'use_reranking': get_env_var('USE_RERANKING', True, bool),
            'bm25_k': get_env_var('BM25_K', 50, int),
            'vector_k': get_env_var('VECTOR_K', 50, int),
            'rerank_k': get_env_var('RERANK_K', 20, int),
            'rrf_k': get_env_var('RRF_K', 60, int),
            'bm25_weight': get_env_var('BM25_WEIGHT', 1.0, float),
            'vector_weight': get_env_var('VECTOR_WEIGHT', 1.0, float)
        }

        # Index config
        config_dict['index'] = {
            'chroma_dir': get_env_var('CHROMA_PERSIST_DIRECTORY', '../data/chroma_db'),
            'chroma_collection': get_env_var('COLLECTION_NAME', 'academic_docs'),
            'cache_dir': get_env_var('INDEX_CACHE_DIR', '../data/cache'),
            'enable_unified_cache': True
        }

        # BM25 config
        config_dict['bm25'] = {
            'k1': get_env_var('bm25_k1', 1.5, float),
            'b': get_env_var('bm25_b', 0.75, float),
            'ngram_range_min': get_env_var('bm25_ngram_min', 1, int),
            'ngram_range_max': get_env_var('bm25_ngram_max', 2, int),
            'min_df': get_env_var('bm25_min_df', 1, int),
            'max_df': get_env_var('bm25_max_df', 0.95, float),
            'sublinear_tf': get_env_var('bm25_sublinear_tf', True, bool)
        }

        # Global settings
        config_dict['log_level'] = get_env_var('log_level', 'INFO')
        config_dict['cache_enabled'] = get_env_var('cache_enabled', True, bool)
        config_dict['timeout_seconds'] = get_env_var('timeout_seconds', 30, int)

        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'embedding': {
                'model_name': self.embedding.model_name,
                'device': self.embedding.device,
                'batch_size': self.embedding.batch_size,
                'max_seq_length': self.embedding.max_seq_length,
                'cache_dir': self.embedding.cache_dir
            },
            'llm': {
                'model_type': self.llm.model_type,
                'temperature': self.llm.temperature,
                'max_tokens': self.llm.max_tokens,
                'top_p': self.llm.top_p,
                'api_key': self.llm.api_key,
                'endpoint': self.llm.endpoint
            },
            'retrieval': {
                'pipeline_type': self.retrieval.pipeline_type,
                'max_results': self.retrieval.max_results,
                'use_reranking': self.retrieval.use_reranking,
                'bm25_k': self.retrieval.bm25_k,
                'vector_k': self.retrieval.vector_k,
                'rerank_k': self.retrieval.rerank_k,
                'rrf_k': self.retrieval.rrf_k,
                'bm25_weight': self.retrieval.bm25_weight,
                'vector_weight': self.retrieval.vector_weight
            },
            'index': {
                'chroma_dir': self.index.chroma_dir,
                'chroma_collection': self.index.chroma_collection,
                'cache_dir': self.index.cache_dir,
                'enable_unified_cache': self.index.enable_unified_cache
            },
            'bm25': {
                'k1': self.bm25.k1,
                'b': self.bm25.b,
                'ngram_range_min': self.bm25.ngram_range_min,
                'ngram_range_max': self.bm25.ngram_range_max,
                'min_df': self.bm25.min_df,
                'max_df': self.bm25.max_df,
                'sublinear_tf': self.bm25.sublinear_tf
            },
            'log_level': self.log_level,
            'cache_enabled': self.cache_enabled,
            'timeout_seconds': self.timeout_seconds
        }

    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        config_dict = self.to_dict()

        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

    def validate(self) -> None:
        """Validate configuration settings."""
        # Validate embedding config
        if not self.embedding.model_name:
            raise ValueError("Embedding model name cannot be empty")

        # Validate LLM config
        if self.llm.model_type not in ["gemini", "ollama"]:
            raise ValueError(f"Invalid LLM type: {self.llm.model_type}. Supported: gemini, ollama")

        if self.llm.needs_api_key and not self.llm.api_key:
            raise ValueError(f"API key required for {self.llm.model_type}")

        # Validate retrieval config
        if self.retrieval.max_results <= 0:
            raise ValueError("max_results must be positive")

        if self.retrieval.pipeline_type not in ["baseline", "advanced"]:
            raise ValueError(f"Invalid pipeline type: {self.retrieval.pipeline_type}")

        # Validate BM25 config
        if self.bm25.k1 <= 0:
            raise ValueError("BM25 k1 parameter must be positive")

        if not (0 <= self.bm25.b <= 1):
            raise ValueError("BM25 b parameter must be between 0 and 1")

        if self.bm25.ngram_range_min < 1 or self.bm25.ngram_range_max < self.bm25.ngram_range_min:
            raise ValueError("Invalid BM25 n-gram range")

        if self.bm25.min_df < 1 or not (0 < self.bm25.max_df <= 1):
            raise ValueError("Invalid BM25 document frequency parameters")

        # Validate index paths
        if not self.index.chroma_dir:
            raise ValueError("ChromaDB path cannot be empty")

        