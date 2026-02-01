"""Tests for LLM Long-Term Memory package."""

import pytest
from unittest.mock import Mock, patch
from llm_memory.backends import (
    Memory,
    SQLiteBackend,
    create_storage_backend,
)


class TestMemory:
    """Tests for Memory dataclass."""
    
    def test_memory_creation(self):
        """Test creating a Memory object."""
        memory = Memory(
            id="test_123",
            content="User prefers dark mode",
            category="preferences",
            importance=0.8,
            timestamp="2024-01-15T10:30:00",
        )
        assert memory.id == "test_123"
        assert memory.content == "User prefers dark mode"
        assert memory.category == "preferences"
        assert memory.importance == 0.8
    
    def test_memory_to_dict(self):
        """Test converting Memory to dictionary."""
        memory = Memory(
            id="test_123",
            content="Test content",
            category="test",
            importance=0.5,
            timestamp="2024-01-15T10:30:00",
        )
        result = memory.to_dict()
        assert isinstance(result, dict)
        assert result["id"] == "test_123"
        assert result["content"] == "Test content"


class TestSQLiteBackend:
    """Tests for SQLite storage backend."""
    
    def test_sqlite_init(self, tmp_path):
        """Test SQLite backend initialization."""
        db_path = str(tmp_path / "test.db")
        backend = SQLiteBackend(db_path=db_path)
        assert backend.db_path == db_path
    
    def test_sqlite_save_and_get(self, tmp_path):
        """Test saving and retrieving a memory."""
        db_path = str(tmp_path / "test.db")
        backend = SQLiteBackend(db_path=db_path)
        
        memory = Memory(
            id="test_123",
            content="Test memory",
            category="test",
            importance=0.8,
            timestamp="2024-01-15T10:30:00",
        )
        
        backend.save_memory(memory)
        retrieved = backend.get_memory("test_123")
        
        assert retrieved is not None
        assert retrieved.id == "test_123"
        assert retrieved.content == "Test memory"
    
    def test_sqlite_delete(self, tmp_path):
        """Test deleting a memory."""
        db_path = str(tmp_path / "test.db")
        backend = SQLiteBackend(db_path=db_path)
        
        memory = Memory(
            id="test_123",
            content="Test memory",
            category="test",
            importance=0.8,
            timestamp="2024-01-15T10:30:00",
        )
        
        backend.save_memory(memory)
        assert backend.delete_memory("test_123") is True
        assert backend.get_memory("test_123") is None
    
    def test_sqlite_get_all(self, tmp_path):
        """Test getting all memories."""
        db_path = str(tmp_path / "test.db")
        backend = SQLiteBackend(db_path=db_path)
        
        for i in range(3):
            memory = Memory(
                id=f"test_{i}",
                content=f"Memory {i}",
                category="test",
                importance=0.5,
                timestamp="2024-01-15T10:30:00",
            )
            backend.save_memory(memory)
        
        all_memories = backend.get_all_memories()
        assert len(all_memories) == 3
    
    def test_sqlite_search(self, tmp_path):
        """Test searching memories."""
        db_path = str(tmp_path / "test.db")
        backend = SQLiteBackend(db_path=db_path)
        
        memories = [
            Memory(id="1", content="I love Python programming", category="preferences", importance=0.8, timestamp=""),
            Memory(id="2", content="I prefer dark mode", category="preferences", importance=0.7, timestamp=""),
            Memory(id="3", content="I use VS Code", category="tools", importance=0.9, timestamp=""),
        ]
        
        for mem in memories:
            backend.save_memory(mem)
        
        results = backend.search_memories("Python")
        assert len(results) == 1
        assert "Python" in results[0].content


class TestCreateStorageBackend:
    """Tests for storage backend factory."""
    
    def test_create_sqlite(self, tmp_path):
        """Test creating SQLite backend via factory."""
        db_path = str(tmp_path / "test.db")
        backend = create_storage_backend("sqlite", db_path=db_path)
        assert isinstance(backend, SQLiteBackend)
    
    def test_invalid_backend_type(self):
        """Test that invalid backend type raises error."""
        with pytest.raises(ValueError):
            create_storage_backend("invalid_backend")
