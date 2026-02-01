# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-01

### Added
- Initial release
- `LongTermMemorySystem` - Main orchestrator for memory operations
- `MemoryExtractor` - Extract memories from conversations using OpenAI GPT
- `VectorStore` - Semantic search with FAISS and OpenAI embeddings
- **Pluggable Storage Backends**:
  - `SQLiteBackend` - Default, no external dependencies
  - `PostgreSQLBackend` - Production-ready with full-text search
  - `MongoDBBackend` - Document-based storage
  - `RedisBackend` - High-performance caching
- `create_storage_backend()` - Factory function for easy backend creation
- Abstract `StorageBackend` interface for custom implementations
- Context manager support for proper connection cleanup
- Comprehensive documentation and examples
- Unit tests for storage backends

### Contributors
- Devendra Parihar ([@Devparihar5](https://github.com/Devparihar5))
- Divya ([@piechartXdata](https://github.com/piechartXdata))
