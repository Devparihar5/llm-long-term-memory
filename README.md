# 🧠 LLM Long-Term Memory

[![PyPI version](https://badge.fury.io/py/llm-long-term-memory.svg)](https://badge.fury.io/py/llm-long-term-memory)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Give your AI agents persistent, searchable long-term memory with pluggable storage backends.**

A sophisticated memory storage and retrieval system that provides LLMs with persistent, searchable long-term memory capabilities. This system can extract, store, update, and retrieve memories from conversations, enabling AI agents to maintain context across multiple sessions.

## ✨ Features

- 🧠 **Intelligent Memory Extraction** - Automatically extracts factual information from conversations using OpenAI GPT
- 🔍 **Semantic Search** - Vector-based similarity search using OpenAI embeddings and FAISS
- 💾 **Pluggable Storage Backends** - SQLite, PostgreSQL, MongoDB, and Redis support
- 🔄 **Memory Management** - Add, update, and delete memories with conflict resolution
- 📊 **Category Organization** - Automatic categorization of memories
- ⚡ **Importance Scoring** - Weighted importance system for memory prioritization
- 🔗 **LangChain Integration** - Built with LangChain for robust LLM interactions

## 📦 Installation

```bash
# Basic installation (SQLite backend)
pip install llm-long-term-memory

# With PostgreSQL support
pip install llm-long-term-memory[postgresql]

# With MongoDB support
pip install llm-long-term-memory[mongodb]

# With Redis support
pip install llm-long-term-memory[redis]

# With all backends
pip install llm-long-term-memory[all]

# With Streamlit UI
pip install llm-long-term-memory[streamlit]
```

## 🚀 Quick Start

```python
from llm_memory import LongTermMemorySystem

# Initialize with SQLite (default)
memory = LongTermMemorySystem(openai_api_key="your-api-key")

# Process a message and extract memories
result = memory.process_message(
    "I use VS Code for Python development and prefer dark mode", 
    user_id="user123"
)
print(f"Extracted {len(result['new_memories'])} memories")

# Query memories
answer = memory.answer_with_memory("What IDE do I use?")
print(answer)  # "You use VS Code for Python development"

# Get all memories
memories = memory.get_all_memories()
for mem in memories:
    print(f"- {mem.content} (importance: {mem.importance})")
```

## 💾 Storage Backends

### SQLite (Default)

```python
from llm_memory import LongTermMemorySystem

memory = LongTermMemorySystem(
    openai_api_key="...",
    storage_backend="sqlite",
    storage_config={"db_path": "my_memories.db"}
)
```

### PostgreSQL

```python
from llm_memory import LongTermMemorySystem

memory = LongTermMemorySystem(
    openai_api_key="...",
    storage_backend="postgresql",
    storage_config={
        "connection_string": "postgresql://user:password@localhost:5432/memory_db"
    }
)
```

### MongoDB

```python
from llm_memory import LongTermMemorySystem

memory = LongTermMemorySystem(
    openai_api_key="...",
    storage_backend="mongodb",
    storage_config={
        "connection_string": "mongodb://localhost:27017",
        "database": "memory_db",
        "collection": "memories"
    }
)
```

### Redis

```python
from llm_memory import LongTermMemorySystem

memory = LongTermMemorySystem(
    openai_api_key="...",
    storage_backend="redis",
    storage_config={
        "host": "localhost",
        "port": 6379,
        "password": "optional_password"
    }
)
```

### Custom Backend

```python
from llm_memory import StorageBackend, Memory, LongTermMemorySystem

class MyCustomBackend(StorageBackend):
    def init_storage(self) -> None:
        # Initialize your storage
        pass
    
    def save_memory(self, memory: Memory) -> None:
        # Save memory
        pass
    
    def get_memory(self, memory_id: str) -> Memory:
        # Get memory by ID
        pass
    
    def get_all_memories(self) -> list:
        # Get all memories
        pass
    
    def delete_memory(self, memory_id: str) -> bool:
        # Delete memory
        pass
    
    def search_memories(self, query: str, category: str = None) -> list:
        # Search memories
        pass
    
    def close(self) -> None:
        # Close connections
        pass

# Use your custom backend
memory = LongTermMemorySystem(
    openai_api_key="...",
    storage_backend=MyCustomBackend()
)
```

## 📊 Memory Structure

```python
from llm_memory import Memory

# Each memory contains:
memory = Memory(
    id="unique_id",
    content="User prefers dark mode",
    category="preferences",
    importance=0.8,
    timestamp="2024-01-15T10:30:00",
    embedding=[...],  # Vector embedding
    metadata={"user_id": "user123", "source": "chat"}
)
```

## 🔧 API Reference

### LongTermMemorySystem

```python
# Initialize
memory = LongTermMemorySystem(
    openai_api_key: str,              # Required: OpenAI API key
    storage_backend: str = "sqlite",   # Backend type or instance
    storage_config: dict = None,       # Backend-specific config
    embedding_model: str = "text-embedding-3-small",
    llm_model: str = "gpt-3.5-turbo",
)

# Methods
memory.process_message(message, user_id, context)  # Extract memories from message
memory.query_memories(query, k=5)                  # Semantic search for memories
memory.answer_with_memory(question, max_memories)  # Answer using memory context
memory.get_all_memories()                          # Get all stored memories
memory.delete_memory(memory_id)                    # Delete specific memory
memory.get_memory_stats()                          # Get statistics
memory.close()                                     # Close connections
```

### Context Manager Support

```python
from llm_memory import LongTermMemorySystem

with LongTermMemorySystem(openai_api_key="...") as memory:
    memory.process_message("Hello!", user_id="user123")
    # Connections automatically closed when done
```

## 🎯 Use Cases

1. **Personal AI Assistants** - Remember user preferences, habits, and information
2. **Customer Service Bots** - Maintain customer history and preferences  
3. **Educational AI** - Track learning progress and personalized content
4. **Productivity Tools** - Remember user workflows and tool preferences
5. **Healthcare AI** - Maintain patient information (with proper security)

## 🌐 Web Interface

If you installed with `[streamlit]`:

```bash
# Clone the repo for the app.py
git clone https://github.com/Devparihar5/llm-long-term-memory.git
cd llm-long-term-memory

# Run the Streamlit app
streamlit run app.py
```

## 🔒 Security Considerations

- Store API keys securely (use environment variables)
- Use secure connection strings for databases
- Consider encryption for sensitive memories
- Implement user authentication for multi-user scenarios

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Vector search powered by [FAISS](https://github.com/facebookresearch/faiss)
- Embeddings by [OpenAI](https://openai.com)

---

**Made with ❤️ by [Devendra Parihar](https://github.com/Devparihar5)**

*Contributions by [Divya](https://github.com/piechartXdata) ✨*
