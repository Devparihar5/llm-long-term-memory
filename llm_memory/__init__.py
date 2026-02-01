"""
LLM Long-Term Memory - A sophisticated memory system for LLM agents.

Give your AI agents persistent, searchable long-term memory with pluggable storage backends.

Basic Usage:
    from llm_memory import LongTermMemorySystem

    memory = LongTermMemorySystem(openai_api_key="...")
    
    # Process messages and extract memories
    memory.process_message("I prefer dark mode and use VS Code", user_id="user123")
    
    # Query memories
    answer = memory.answer_with_memory("What IDE do I use?")

With PostgreSQL:
    memory = LongTermMemorySystem(
        openai_api_key="...",
        storage_backend="postgresql",
        storage_config={"connection_string": "postgresql://user:pass@localhost/db"}
    )

With MongoDB:
    memory = LongTermMemorySystem(
        openai_api_key="...",
        storage_backend="mongodb",
        storage_config={"connection_string": "mongodb://localhost:27017", "database": "memory_db"}
    )

With Redis:
    memory = LongTermMemorySystem(
        openai_api_key="...",
        storage_backend="redis",
        storage_config={"host": "localhost", "port": 6379}
    )
"""

__version__ = "0.1.0"
__author__ = "Devendra Parihar"
__email__ = "dev@devendraparihar.com"

# Core classes
from llm_memory.core import (
    LongTermMemorySystem,
    MemoryExtractor,
    VectorStore,
)

# Storage backends
from llm_memory.backends import (
    Memory,
    StorageBackend,
    SQLiteBackend,
    PostgreSQLBackend,
    MongoDBBackend,
    RedisBackend,
    create_storage_backend,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "LongTermMemorySystem",
    "MemoryExtractor", 
    "VectorStore",
    # Data classes
    "Memory",
    # Backends
    "StorageBackend",
    "SQLiteBackend",
    "PostgreSQLBackend",
    "MongoDBBackend",
    "RedisBackend",
    "create_storage_backend",
]
