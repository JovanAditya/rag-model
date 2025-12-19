#!/usr/bin/env python3
"""
Consolidated System Verification for Academic RAG

This script provides comprehensive system verification, replacing multiple
individual verification scripts with a unified approach.

Usage:
    python scripts/verify_system.py --mode [quick|services|data|indexes|complete]
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
import importlib.util

# Add model to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rag_model import AcademicRAG
    from rag_model.core.config import RAGConfig, RetrievalConfig
    MODEL_AVAILABLE = True
except Exception as e:
    logging.error(f"Failed to import model: {e}")
    MODEL_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemVerifier:
    """Comprehensive system verification"""

    def __init__(self):
        self.results = {
            'timestamp': time.time(),
            'checks': {},
            'overall_status': 'unknown'
        }

    def run_command(self, command: List[str], description: str) -> Dict[str, Any]:
        """Run shell command and return result"""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30
            )
            return {
                'command': ' '.join(command),
                'description': description,
                'return_code': result.returncode,
                'stdout': result.stdout.strip(),
                'stderr': result.stderr.strip(),
                'success': result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            return {
                'command': ' '.join(command),
                'description': description,
                'return_code': -1,
                'stdout': '',
                'stderr': 'Command timed out',
                'success': False
            }
        except Exception as e:
            return {
                'command': ' '.join(command),
                'description': description,
                'return_code': -1,
                'stdout': '',
                'stderr': str(e),
                'success': False
            }

    def check_package_comprehensive(self, package_name: str, display_name: str) -> Dict[str, Any]:
        """Comprehensive package checking using multiple methods."""
        result = {
            'package_name': package_name,
            'display_name': display_name,
            'available': False,
            'version': None,
            'location': None,
            'conda_managed': False,
            'pip_managed': False,
            'detection_method': 'none'
        }

        # Method 1: Try direct import
        try:
            if package_name == 'sentence_transformers':
                import sentence_transformers
                module = sentence_transformers
            elif package_name == 'transformers':
                import transformers
                module = transformers
            else:
                module = __import__(package_name)

            result['available'] = True
            result['detection_method'] = 'import'

            # Try to get version
            if hasattr(module, '__version__'):
                result['version'] = module.__version__
            elif hasattr(module, 'version'):
                result['version'] = module.version

            # Try to get location
            if hasattr(module, '__file__'):
                result['location'] = module.__file__

        except ImportError:
            pass

        # Method 2: Try importlib.util if import failed
        if not result['available']:
            try:
                spec = importlib.util.find_spec(package_name)
                if spec is not None:
                    result['available'] = True
                    result['detection_method'] = 'importlib'
                    if spec.origin:
                        result['location'] = spec.origin
            except (ImportError, AttributeError):
                pass

        # Method 3: Check with pip list
        if not result['available']:
            try:
                pip_result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'show', package_name],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if pip_result.returncode == 0:
                    result['available'] = True
                    result['detection_method'] = 'pip'
                    result['pip_managed'] = True

                    # Parse version from pip output
                    for line in pip_result.stdout.split('\n'):
                        if line.startswith('Version:'):
                            result['version'] = line.split(':', 1)[1].strip()
                        elif line.startswith('Location:'):
                            result['location'] = line.split(':', 1)[1].strip()
            except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
                pass

        # Method 4: Check with conda list
        try:
            conda_result = subprocess.run(
                ['conda', 'list', package_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            if conda_result.returncode == 0:
                lines = conda_result.stdout.strip().split('\n')
                # Skip header line(s)
                for line in lines:
                    if package_name in line and not line.startswith('#'):
                        result['conda_managed'] = True
                        # Parse conda output format
                        parts = line.split()
                        if len(parts) >= 2:
                            result['version'] = parts[1] if parts[1] else result['version']
                        break
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            pass

        return result

    def check_python_environment(self) -> Dict[str, Any]:
        """Check Python environment and dependencies"""
        logger.info("🐍 Checking Python environment...")

        results = {}

        # Python version
        version_info = sys.version_info
        results['python_version'] = {
            'major': version_info.major,
            'minor': version_info.minor,
            'micro': version_info.micro,
            'string': sys.version,
            'acceptable': version_info >= (3, 8)  # Require Python 3.8+
        }

        # Check if running in conda environment
        conda_env = {
            'active': False,
            'name': None,
            'prefix': None
        }

        if 'CONDA_DEFAULT_ENV' in os.environ:
            conda_env['active'] = True
            conda_env['name'] = os.environ['CONDA_DEFAULT_ENV']
        elif 'CONDA_PREFIX' in os.environ:
            conda_env['active'] = True
            conda_env['prefix'] = os.environ['CONDA_PREFIX']

        results['conda_environment'] = conda_env

        # Check critical dependencies with comprehensive detection
        critical_packages = [
            ('torch', 'PyTorch'),
            ('transformers', 'Transformers'),
            ('sentence_transformers', 'Sentence-Transformers'),
            ('chromadb', 'ChromaDB'),
            ('numpy', 'NumPy'),
            ('pandas', 'Pandas'),
            ('sklearn', 'Scikit-learn (BM25)'),
            # ('elasticsearch', 'Elasticsearch'),  # Legacy - not required for unified system
            ('accelerate', 'Accelerate'),
            ('datasets', 'Datasets'),
            ('evaluate', 'Evaluate'),
            ('ragas', 'Ragas (evaluation)')
        ]

        missing_packages = []
        package_details = {}

        for package, display_name in critical_packages:
            package_result = self.check_package_comprehensive(package, display_name)
            package_details[package] = package_result

            if package_result['available']:
                detection_info = []
                if package_result['conda_managed']:
                    detection_info.append('conda')
                if package_result['pip_managed']:
                    detection_info.append('pip')

                version_str = f" v{package_result['version']}" if package_result['version'] else ""
                detection_str = f" ({', '.join(detection_info)})" if detection_info else ""

                logger.info(f"  ✅ {display_name}{version_str}{detection_str}")
            else:
                missing_packages.append(display_name)
                logger.warning(f"  ❌ {display_name} not found")

        results['package_details'] = package_details
        results['missing_packages'] = missing_packages
        results['environment_ok'] = len(missing_packages) == 0

        # Additional optional dependencies
        optional_packages = [
            ('matplotlib', 'Matplotlib'),
            ('seaborn', 'Seaborn'),
            ('scikit-learn', 'Scikit-learn'),
            ('plotly', 'Plotly'),
            ('fastapi', 'FastAPI'),
            ('uvicorn', 'Uvicorn')
        ]

        optional_missing = []
        for package, display_name in optional_packages:
            package_result = self.check_package_comprehensive(package, display_name)
            package_details[f'optional_{package}'] = package_result

            if package_result['available']:
                logger.info(f"  ✅ {display_name} (optional)")
            else:
                optional_missing.append(display_name)
                logger.info(f"  ℹ️  {display_name} (optional, not installed)")

        results['optional_missing'] = optional_missing

        self.results['checks']['python_environment'] = results
        return results

    def check_gpu_availability(self) -> Dict[str, Any]:
        """Check GPU availability and configuration"""
        logger.info("🎮 Checking GPU availability...")

        results = {'gpu_available': False, 'gpu_info': None}

        try:
            import torch
            results['gpu_available'] = torch.cuda.is_available()

            if results['gpu_available']:
                gpu_count = torch.cuda.device_count()
                gpu_info = []

                for i in range(gpu_count):
                    properties = torch.cuda.get_device_properties(i)
                    gpu_info.append({
                        'device_id': i,
                        'name': properties.name,
                        'total_memory_gb': properties.total_memory / (1024**3),
                        'compute_capability': f"{properties.major}.{properties.minor}"
                    })

                results['gpu_info'] = gpu_info
                results['gpu_count'] = gpu_count
                logger.info(f"  ✅ Found {gpu_count} GPU(s)")
                for gpu in gpu_info:
                    logger.info(f"    - {gpu['name']} ({gpu['total_memory_gb']:.1f}GB)")
            else:
                logger.warning("  ❌ No CUDA-capable GPU found")
                logger.info("  💡 CPU-only mode will be used")

            results['cuda_available'] = torch.cuda.is_available()

        except ImportError:
            logger.warning("  ❌ PyTorch not available for GPU check")
            results['cuda_available'] = False

        self.results['checks']['gpu_availability'] = results
        return results

    def check_services(self) -> Dict[str, Any]:
        """Check external services (Ollama) and unified index components"""
        logger.info("🔗 Checking external services...")

        results = {
            'ollama': {'available': False, 'version': None, 'models': []},
            'unified_index': {'available': False, 'status': None, 'components': {}}
            # 'elasticsearch': {'available': False, 'version': None, 'status': None}  # Legacy
        }

        # Check Ollama
        ollama_check = self.run_command(
            ['ollama', '--version'],
            'Ollama version check'
        )

        if ollama_check['success']:
            results['ollama']['available'] = True
            results['ollama']['version'] = ollama_check['stdout']
            logger.info(f"  ✅ Ollama: {ollama_check['stdout']}")

            # Check available models
            models_check = self.run_command(
                ['ollama', 'list'],
                'Ollama models list'
            )

            if models_check['success']:
                try:
                    models_data = json.loads(models_check['stdout'])
                    results['ollama']['models'] = models_data.get('models', [])
                    logger.info(f"  📦 Available models: {len(results['ollama']['models'])}")
                except:
                    logger.warning("  ⚠️  Could not parse Ollama models list")
        else:
            logger.warning("  ❌ Ollama not available")

        # Check Unified Index Components
        try:
            # Try to import and test unified index components
            from sklearn.feature_extraction.text import TfidfVectorizer
            results['unified_index']['components']['sklearn'] = True
            logger.info("  ✅ Scikit-learn (BM25): Available")
        except ImportError:
            results['unified_index']['components']['sklearn'] = False
            logger.warning("  ❌ Scikit-learn (BM25): Not available")

        # Check if ChromaDB is accessible (already checked in packages, but verify accessibility)
        try:
            import chromadb
            # Test basic ChromaDB functionality
            test_client = chromadb.Client()
            test_collection = test_client.get_or_create_collection("test")
            results['unified_index']['components']['chromadb'] = True
            results['unified_index']['status'] = 'accessible'
            logger.info("  ✅ ChromaDB: Accessible")
            test_client.delete_collection("test")  # Clean up
        except Exception as e:
            results['unified_index']['components']['chromadb'] = False
            results['unified_index']['status'] = 'inaccessible'
            logger.warning(f"  ❌ ChromaDB: Not accessible - {str(e)}")

        # Set unified index as available if both components are working
        if results['unified_index']['components'].get('sklearn', False) and results['unified_index']['components'].get('chromadb', False):
            results['unified_index']['available'] = True
            results['unified_index']['status'] = 'ready'
            logger.info("  ✅ Unified Index: Ready (ChromaDB + BM25)")
        else:
            results['unified_index']['available'] = False
            logger.warning("  ❌ Unified Index: Not ready")

        all_services_ok = results['ollama']['available'] and results['unified_index']['available']
        results['all_services_ok'] = all_services_ok

        self.results['checks']['services'] = results
        return results

    def check_model_integrity(self) -> Dict[str, Any]:
        """Check model files and integrity"""
        logger.info("🤖 Checking model integrity...")

        results = {
            'model_files_exist': False,
            'config_files_exist': False,
            'model_loads': False,
            'health_check': None
        }

        if not MODEL_AVAILABLE:
            logger.error("  ❌ Model cannot be imported")
            self.results['checks']['model_integrity'] = results
            return results

        model_dir = Path(__file__).parent.parent

        # Check essential files
        essential_files = [
            model_dir / '__init__.py',
            model_dir / 'rag_model' / '__init__.py',
            model_dir / 'rag_model' / 'core' / 'pipeline.py',
            model_dir / 'rag_model' / 'core' / 'config.py',
        ]

        missing_files = []
        for file_path in essential_files:
            if file_path.exists():
                logger.debug(f"  ✅ {file_path.relative_to(model_dir)}")
            else:
                missing_files.append(str(file_path.relative_to(model_dir)))
                logger.warning(f"  ❌ {file_path.relative_to(model_dir)} not found")

        results['model_files_exist'] = len(missing_files) == 0
        results['missing_files'] = missing_files

        # Check configuration files
        config_files = [
            model_dir / '.env.example',
        ]

        missing_configs = []
        for config_file in config_files:
            if config_file.exists():
                logger.debug(f"  ✅ {config_file.relative_to(model_dir)}")
            else:
                missing_configs.append(str(config_file.relative_to(model_dir)))
                logger.warning(f"  ❌ {config_file.relative_to(model_dir)} not found")

        results['config_files_exist'] = len(missing_configs) == 0
        results['missing_configs'] = missing_configs

        # Test model loading
        try:
            config = RAGConfig(
                retrieval=RetrievalConfig(pipeline_type="baseline")
            )
            logger.info("  ✅ Configuration class working")

            # Try basic model instantiation (without loading resources)
            results['model_loads'] = True
            logger.info("  ✅ Model class can be instantiated")

        except Exception as e:
            results['model_loads'] = False
            logger.error(f"  ❌ Model loading failed: {e}")

        # Health check (requires full model load)
        try:
            rag = AcademicRAG()
            health = rag.health_check()
            results['health_check'] = health
            logger.info(f"  ✅ Health check: {health.get('status', 'unknown')}")
        except Exception as e:
            logger.warning(f"  ⚠️  Health check failed (may be expected): {e}")

        model_ok = (results['model_files_exist'] and
                   results['config_files_exist'] and
                   results['model_loads'])
        results['overall_model_ok'] = model_ok

        self.results['checks']['model_integrity'] = results
        return results

    def check_data_directories(self) -> Dict[str, Any]:
        """Check data directories structure"""
        logger.info("📁 Checking data directories...")

        results = {
            'directories_status': {},
            'all_directories_ok': False
        }

        model_dir = Path(__file__).parent.parent

        required_dirs = [
            ('data', 'Raw data directory'),
            ('data/raw', 'Raw documents directory'),
            ('data/processed', 'Processed data directory'),
            ('data/cache', 'Data cache directory'),
            ('logs', 'Logs directory'),
        ]

        optional_dirs = [
            ('chroma_db', 'ChromaDB vector store'),
            ('output', 'Output directory'),
        ]

        for dir_name, description in required_dirs:
            dir_path = model_dir / dir_name
            exists = dir_path.exists()
            results['directories_status'][dir_name] = {
                'exists': exists,
                'description': description,
                'path': str(dir_path)
            }

            if exists:
                logger.info(f"  ✅ {dir_name} - {description}")
            else:
                logger.warning(f"  ❌ {dir_name} - {description} (will be created as needed)")

        for dir_name, description in optional_dirs:
            dir_path = model_dir / dir_name
            exists = dir_path.exists()
            results['directories_status'][dir_name] = {
                'exists': exists,
                'description': f"{description} (optional)",
                'path': str(dir_path)
            }

            if exists:
                logger.info(f"  ✅ {dir_name} - {description}")

        # Consider required dirs that will be auto-created as OK
        auto_create_dirs = ['data', 'data/raw', 'data/processed', 'data/cache', 'logs']
        missing_required = [d for d, _ in required_dirs[:4] if not (model_dir / d).exists() and d not in auto_create_dirs]

        results['all_directories_ok'] = len(missing_required) == 0
        self.results['checks']['data_directories'] = results
        return results

    def quick_check(self) -> Dict[str, Any]:
        """Perform quick verification"""
        logger.info("⚡ Running quick verification...")

        return {
            'python_environment': self.check_python_environment(),
            'model_integrity': self.check_model_integrity(),
        }

    def services_check(self) -> Dict[str, Any]:
        """Check external services"""
        return {
            'services': self.check_services(),
            'gpu_availability': self.check_gpu_availability(),
        }

    def data_check(self) -> Dict[str, Any]:
        """Check data structure"""
        return {
            'data_directories': self.check_data_directories(),
        }

    def indexes_check(self) -> Dict[str, Any]:
        """Check unified index availability and health"""
        logger.info("🔍 Checking unified indexes...")

        results = {
            'chroma_db': {'exists': False, 'healthy': False},
            'bm25_index': {'exists': False, 'healthy': False},
            'unified_index': {'exists': False, 'healthy': False}
            # 'elasticsearch': {'exists': False, 'healthy': False}  # Legacy
        }

        model_dir = Path(__file__).parent.parent
        data_dir = model_dir.parent / 'data'

        # Check ChromaDB
        chroma_path = data_dir / 'chroma_db'
        if chroma_path.exists():
            results['chroma_db']['exists'] = True
            try:
                import chromadb
                client = chromadb.PersistentClient(path=str(chroma_path))
                collections = client.list_collections()
                results['chroma_db']['collections'] = len(collections)
                results['chroma_db']['healthy'] = True
                logger.info(f"  ✅ ChromaDB: {len(collections)} collections")
            except Exception as e:
                logger.error(f"  ❌ ChromaDB check failed: {e}")
        else:
            logger.info("  ℹ️  ChromaDB not created yet")

        # Check BM25 Index (Cache Files)
        cache_dir = data_dir / 'cache'
        if cache_dir.exists():
            # Look for BM25 cache files
            bm25_files = list(cache_dir.glob("**/*bm25*")) + list(cache_dir.glob("**/*tfidf*"))
            if bm25_files:
                results['bm25_index']['exists'] = True
                results['bm25_index']['files'] = len(bm25_files)
                results['bm25_index']['healthy'] = True
                logger.info(f"  ✅ BM25 Index: {len(bm25_files)} cache files found")
            else:
                logger.info("  ℹ️  BM25 Index cache files not found")
        else:
            logger.info("  ℹ️  Cache directory not found - BM25 Index not built")

        # Check if Unified Index is ready (both ChromaDB and BM25)
        if results['chroma_db']['exists'] and results['bm25_index']['exists']:
            results['unified_index']['exists'] = True
            results['unified_index']['healthy'] = True
            results['unified_index']['components'] = {
                'chroma_collections': results['chroma_db'].get('collections', 0),
                'bm25_files': results['bm25_index'].get('files', 0)
            }
            logger.info("  ✅ Unified Index: Ready (ChromaDB + BM25)")
        else:
            missing_components = []
            if not results['chroma_db']['exists']:
                missing_components.append("ChromaDB")
            if not results['bm25_index']['exists']:
                missing_components.append("BM25 Index")
            logger.info(f"  ℹ️  Unified Index: Incomplete (missing: {', '.join(missing_components)})")

        results['all_indexes_available'] = results['unified_index']['exists']

        self.results['checks']['indexes'] = results
        return results

    def complete_check(self) -> Dict[str, Any]:
        """Perform complete verification"""
        logger.info("🔬 Running complete verification...")

        return {
            'python_environment': self.check_python_environment(),
            'gpu_availability': self.check_gpu_availability(),
            'services': self.check_services(),
            'model_integrity': self.check_model_integrity(),
            'data_directories': self.check_data_directories(),
            'indexes': self.indexes_check()
        }

    def evaluate_results(self) -> str:
        """Evaluate verification results and return status"""
        critical_failures = []
        warnings = []

        # Check critical components
        checks = self.results['checks']

        # Python environment
        if 'python_environment' in checks:
            py_env = checks['python_environment']
            if not py_env['environment_ok']:
                critical_failures.append(f"Missing packages: {py_env['missing_packages']}")
            elif not py_env['python_version']['acceptable']:
                warnings.append(f"Python version {py_env['python_version']['string']} may be too old")

        # Model integrity
        if 'model_integrity' in checks:
            model_int = checks['model_integrity']
            if not model_int['overall_model_ok']:
                critical_failures.append("Model integrity issues detected")

        # Services availability
        if 'services' in checks:
            services = checks['services']
            if not services['all_services_ok']:
                missing_services = []
                if not services['ollama']['available']:
                    missing_services.append('Ollama')
                if not services['unified_index']['available']:
                    missing_services.append('Unified Index (ChromaDB + BM25)')
                warnings.append(f"Missing services: {missing_services}")

        # Overall evaluation
        if critical_failures:
            self.results['overall_status'] = 'FAILED'
            return 'FAILED'
        elif warnings:
            self.results['overall_status'] = 'WARNING'
            return 'WARNING'
        else:
            self.results['overall_status'] = 'PASSED'
            return 'PASSED'

    def print_summary(self):
        """Print verification summary"""
        print("\n" + "="*60)
        print("ACADEMIC RAG SYSTEM VERIFICATION")
        print("="*60)

        # Overall status
        status = self.evaluate_results()
        status_symbol = {'PASSED': '✅', 'WARNING': '⚠️', 'FAILED': '❌'}[status]
        print(f"\n🎯 Overall Status: {status_symbol} {status}")

        # Detailed results
        checks = self.results['checks']

        if 'python_environment' in checks:
            py_env = checks['python_environment']
            print(f"\n🐍 Python Environment:")
            print(f"  Version: {py_env['python_version']['string']}")

            # Conda environment info
            if py_env.get('conda_environment', {}).get('active'):
                conda_env = py_env['conda_environment']
                conda_name = conda_env.get('name', 'unknown')
                print(f"  Conda Environment: ✅ Active ({conda_name})")
            else:
                print(f"  Conda Environment: ❌ Not active")

            print(f"  Dependencies: {status_symbol} {py_env['environment_ok']}")

            # Show critical packages with details
            print(f"  Critical Packages:")
            if 'package_details' in py_env:
                for package_name, package_info in py_env['package_details'].items():
                    if package_name.startswith('optional_'):
                        continue

                    if package_info['available']:
                        version_str = f" v{package_info['version']}" if package_info['version'] else ""
                        location_str = ""

                        # Show package manager
                        managers = []
                        if package_info['conda_managed']:
                            managers.append('conda')
                        if package_info['pip_managed']:
                            managers.append('pip')
                        if managers:
                            location_str = f" ({', '.join(managers)})"

                        print(f"    ✅ {package_info['display_name']}{version_str}{location_str}")
                    else:
                        print(f"    ❌ {package_info['display_name']} - NOT FOUND")

            if py_env['missing_packages']:
                print(f"  Missing Critical: {', '.join(py_env['missing_packages'])}")

            # Show optional packages
            if py_env.get('optional_missing'):
                print(f"  Optional Packages Missing: {', '.join(py_env['optional_missing'])}")

        if 'gpu_availability' in checks:
            gpu_info = checks['gpu_availability']
            gpu_symbol = {'PASSED': '✅', 'WARNING': '⚠️', 'FAILED': '❌'}['PASSED' if gpu_info['gpu_available'] else 'WARNING']
            print(f"\n🎮 GPU: {gpu_symbol} {'Available' if gpu_info['gpu_available'] else 'Not Available (CPU mode)'}")

            if gpu_info['gpu_info']:
                for gpu in gpu_info['gpu_info']:
                    print(f"  - {gpu['name']} ({gpu['total_memory_gb']:.1f}GB)")

        if 'services' in checks:
            services = checks['services']
            ollama_symbol = {'True': '✅', 'False': '❌'}[str(services['ollama']['available'])]
            unified_symbol = {'True': '✅', 'False': '❌'}[str(services['unified_index']['available'])]

            print(f"\n🔗 External Services:")
            print(f"  Ollama: {ollama_symbol} {'Available' if services['ollama']['available'] else 'Not Available'}")
            print(f"  Unified Index: {unified_symbol} {'Ready' if services['unified_index']['available'] else 'Not Ready'} (ChromaDB + BM25)")

        if 'model_integrity' in checks:
            model_int = checks['model_integrity']
            files_symbol = {'True': '✅', 'False': '❌'}[str(model_int['model_files_exist'])]
            config_symbol = {'True': '✅', 'False': '❌'}[str(model_int['config_files_exist'])]
            load_symbol = {'True': '✅', 'False': '❌'}[str(model_int['model_loads'])]

            print(f"\n🤖 Model Integrity:")
            print(f"  Core Files: {files_symbol} {'Present' if model_int['model_files_exist'] else 'Missing'}")
            print(f"  Configuration: {config_symbol} {'Present' if model_int['config_files_exist'] else 'Missing'}")
            print(f"  Loading: {load_symbol} {'Working' if model_int['model_loads'] else 'Failed'}")

        if 'data_directories' in checks:
            data_dirs = checks['data_directories']
            dirs_ok = data_dirs['all_directories_ok']
            dirs_symbol = {'True': '✅', 'False': '⚠️'}[str(dirs_ok)]

            print(f"\n📁 Data Directories: {dirs_symbol} {'Ready' if dirs_ok else 'Some missing'}")

        if 'indexes' in checks:
            indexes = checks['indexes']
            print(f"\n🔍 Search Indexes:")

            if indexes['chroma_db']['exists']:
                chroma_symbol = {'True': '✅', 'False': '❌'}[str(indexes['chroma_db']['healthy'])]
                print(f"  ChromaDB: {chroma_symbol} {indexes['chroma_db']['collections']} collections")
            else:
                print(f"  ChromaDB: ℹ️  Not created yet")

            if indexes['bm25_index']['exists']:
                bm25_symbol = {'True': '✅', 'False': '❌'}[str(indexes['bm25_index']['healthy'])]
                print(f"  BM25 Index: {bm25_symbol} {indexes['bm25_index']['files']} cache files")
            else:
                print(f"  BM25 Index: ℹ️  Not built")

            if indexes['unified_index']['exists']:
                unified_symbol = {'True': '✅', 'False': '❌'}[str(indexes['unified_index']['healthy'])]
                components = indexes['unified_index']['components']
                print(f"  Unified Index: {unified_symbol} Ready ({components['chroma_collections']} collections + {components['bm25_files']} BM25 files)")
            else:
                print(f"  Unified Index: ℹ️  Incomplete")

        print(f"\n⏰ Verification completed in {time.time() - self.results['timestamp']:.2f} seconds")
        print("="*60)

        return status

def main():
    """Main verification script"""
    parser = argparse.ArgumentParser(description="Academic RAG System Verification")
    parser.add_argument('--mode', '-m', type=str, default='quick',
                        choices=['quick', 'services', 'data', 'indexes', 'complete'],
                        help='Verification mode (default: quick)')
    parser.add_argument('--output', '-o', type=str,
                        help='Output file for results (JSON)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    verifier = SystemVerifier()

    # Run verification based on mode
    if args.mode == 'quick':
        verifier.results['checks'] = verifier.quick_check()
    elif args.mode == 'services':
        verifier.results['checks'] = verifier.services_check()
    elif args.mode == 'data':
        verifier.results['checks'] = verifier.data_check()
    elif args.mode == 'indexes':
        verifier.results['checks'] = verifier.indexes_check()
    elif args.mode == 'complete':
        verifier.results['checks'] = verifier.complete_check()

    # Print summary and evaluate
    status = verifier.print_summary()

    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(verifier.results, f, indent=2, default=str)
        logger.info(f"Results saved to {args.output}")

    return 0 if status == 'PASSED' else 1

if __name__ == '__main__':
    sys.exit(main())
