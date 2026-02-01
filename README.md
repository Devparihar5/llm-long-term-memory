# 🧠 Advanced Long-Term Memory System for LLM Agents

A sophisticated memory storage and retrieval system that provides LLMs with persistent, searchable long-term memory capabilities. This system can extract, store, update, and retrieve memories from conversations, enabling AI agents to maintain context across multiple sessions.

## ✨ Features

- **Intelligent Memory Extraction**: Automatically extracts factual information from conversations using OpenAI GPT
- **Semantic Search**: Vector-based similarity search using OpenAI embeddings and FAISS
- **Memory Management**: Add, update, and delete memories with conflict resolution
- **🆕 Pluggable Storage Backends**: SQLite, PostgreSQL, MongoDB, and Redis support
- **Category Organization**: Automatic categorization of memories (tools, preferences, personal, habits, etc.)
- **Importance Scoring**: Weighted importance system for memory prioritization
- **Real-time Updates**: Detect and process memory updates and deletions from natural language
- **Web Interface**: Comprehensive Streamlit-based testing and management interface
- **LangChain Integration**: Built with LangChain for robust LLM interactions
- **Modular Architecture**: Clean separation of concerns with well-defined components

## 🏗️ Architecture

The system follows a layered architecture with clear separation of concerns:

![System Architecture](media/memory_architecture.png)

### Core Components

1. **LongTermMemorySystem** - Main orchestrator that coordinates all components
2. **MemoryExtractor** - Uses OpenAI GPT via LangChain to extract and categorize memories
3. **VectorStore** - Handles embedding generation and semantic search using OpenAI embeddings and FAISS
4. **StorageBackend** - Pluggable storage layer with multiple backend options

### Storage Backends

| Backend | Best For | Install |
|---------|----------|---------|
| **SQLite** (default) | Development, single-user | Built-in |
| **PostgreSQL** | Production, multi-user, full-text search | `pip install psycopg2-binary` |
| **MongoDB** | Document-based, flexible schema | `pip install pymongo` |
| **Redis** | High-performance caching, fast access | `pip install redis` |

## 📁 Project Structure

```
llm-long-term-memory/
├── 📄 memory_system.py          # Core memory system implementation
├── 📄 storage_backends.py       # Pluggable storage backends
├── 🌐 app.py                    # Streamlit web interface
├── 📋 requirements.txt          # Python dependencies
├── 📖 README.md                 # Project documentation
├── 📁 media/                    # Documentation assets
```

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- Required packages (see requirements.txt)
- Optional: PostgreSQL, MongoDB, or Redis for alternative storage

## 🚀 Installation

1. Clone or download the project files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Install additional storage backends:
   ```bash
   pip install psycopg2-binary  # PostgreSQL
   pip install pymongo          # MongoDB
   pip install redis            # Redis
   ```

## 💡 Usage

### Basic Usage (SQLite - Default)

```python
from memory_system import LongTermMemorySystem

memory_system = LongTermMemorySystem(openai_api_key="your-api-key")

# Process a message and extract memories
result = memory_system.process_message(
    "I use VS Code for Python development", 
    user_id="user123"
)

# Query memories
answer = memory_system.answer_with_memory(
    "What IDE do I use?"
)
print(answer)  # Output: "You use VS Code for Python development"
```

### 🆕 Using PostgreSQL Backend

```python
from memory_system import LongTermMemorySystem

memory_system = LongTermMemorySystem(
    openai_api_key="your-api-key",
    storage_backend="postgresql",
    storage_config={
        "connection_string": "postgresql://user:password@localhost:5432/memory_db"
    }
)

# Works exactly the same as SQLite!
result = memory_system.process_message("I prefer dark mode", user_id="user123")
```

### 🆕 Using MongoDB Backend

```python
from memory_system import LongTermMemorySystem

memory_system = LongTermMemorySystem(
    openai_api_key="your-api-key",
    storage_backend="mongodb",
    storage_config={
        "connection_string": "mongodb://localhost:27017",
        "database": "memory_db",
        "collection": "memories"
    }
)
```

### 🆕 Using Redis Backend

```python
from memory_system import LongTermMemorySystem

memory_system = LongTermMemorySystem(
    openai_api_key="your-api-key",
    storage_backend="redis",
    storage_config={
        "host": "localhost",
        "port": 6379,
        "password": "optional_password"  # optional
    }
)
```

### 🆕 Using Custom Backend Instance

```python
from memory_system import LongTermMemorySystem
from storage_backends import PostgreSQLBackend

# Create and configure your own backend
custom_backend = PostgreSQLBackend(
    connection_string="postgresql://user:pass@localhost/db"
)

memory_system = LongTermMemorySystem(
    openai_api_key="your-api-key",
    storage_backend=custom_backend  # Pass instance directly
)
```

### 🆕 Context Manager Support

```python
from memory_system import LongTermMemorySystem

# Automatically closes connections when done
with LongTermMemorySystem(
    openai_api_key="your-api-key",
    storage_backend="postgresql",
    storage_config={"connection_string": "..."}
) as memory_system:
    result = memory_system.process_message("Hello!", user_id="user123")
```

### Memory Operations

```python
# Extract memories from conversation
result = memory_system.process_message(
    "I switched from VS Code to NeoVim",
    user_id="user123"
)

# Search for similar memories
memories = memory_system.query_memories("text editors", k=5)

# Get all memories
all_memories = memory_system.get_all_memories()

# Delete a specific memory
memory_system.delete_memory(memory_id)

# Get memory statistics (now includes storage backend info!)
stats = memory_system.get_memory_stats()
# {'total_memories': 10, 'categories': {...}, 'storage_backend': 'PostgreSQLBackend'}
```

### Web Interface

Launch the Streamlit interface:

```bash
streamlit run app.py
```

The web interface provides:
- **Chat & Memory**: Interactive conversation with memory extraction
- **Query Memories**: Search and question-answering interface
- **Memory Analytics**: Visualizations and statistics
- **Memory Management**: View, filter, and delete memories

## 📊 Memory Structure

Each memory contains:

```python
@dataclass
class Memory:
    id: str                    # Unique identifier
    content: str              # The actual memory content
    category: str             # Category (tools, preferences, personal, etc.)
    importance: float         # Importance score (0.0 to 1.0)
    timestamp: str           # Creation/update timestamp
    embedding: List[float]   # Vector embedding for semantic search
    metadata: Dict          # Additional metadata (user_id, source, etc.)
```

## 🆕 Storage Backend API

Create your own custom backend by implementing the `StorageBackend` interface:

```python
from storage_backends import StorageBackend, Memory

class MyCustomBackend(StorageBackend):
    def init_storage(self) -> None:
        """Initialize storage (create tables, etc.)"""
        pass
    
    def save_memory(self, memory: Memory) -> None:
        """Save a memory"""
        pass
    
    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get memory by ID"""
        pass
    
    def get_all_memories(self) -> List[Memory]:
        """Get all memories"""
        pass
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory"""
        pass
    
    def search_memories(self, query: str, category: str = None) -> List[Memory]:
        """Search memories"""
        pass
    
    def close(self) -> None:
        """Close connections"""
        pass
```

## 🎯 Use Cases

1. **Personal AI Assistants**: Remember user preferences, habits, and information
2. **Customer Service Bots**: Maintain customer history and preferences
3. **Educational AI**: Track learning progress and personalized content
4. **Productivity Tools**: Remember user workflows and tool preferences
5. **Healthcare AI**: Maintain patient information and medical history (with proper security)

## 🔒 Security Considerations

- **API Key Security**: Store OpenAI API keys securely (use environment variables)
- **Database Credentials**: Use secure connection strings, avoid hardcoding passwords
- **Data Privacy**: Consider encryption for sensitive memories
- **Access Control**: Implement user authentication for multi-user scenarios
- **Data Retention**: Implement memory expiration policies if needed

## 📝 API Reference

### LongTermMemorySystem

Main class for memory operations:

| Method | Description |
|--------|-------------|
| `process_message(message, user_id, context)` | Extract memories from message |
| `query_memories(query, k)` | Search for similar memories |
| `answer_with_memory(question, max_memories)` | Answer using memory context |
| `get_all_memories()` | Retrieve all stored memories |
| `delete_memory(memory_id)` | Delete specific memory |
| `get_memory_stats()` | Get system statistics |
| `close()` | Close storage connections |

### Constructor Options

```python
LongTermMemorySystem(
    openai_api_key: str,              # Required: OpenAI API key
    storage_backend: str = "sqlite",   # Backend type or instance
    storage_config: dict = None,       # Backend-specific config
    embedding_model: str = "text-embedding-3-small",
    llm_model: str = "gpt-3.5-turbo",
    db_path: str = None               # Legacy: SQLite path
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - feel free to use in your projects!

---

**Happy LLM Memory Building! 🧠✨**

*Contributed by [Divya](https://github.com/piechartXdata) - AI companion to [@Devparihar5](https://github.com/Devparihar5)*
