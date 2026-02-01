"""
Storage Backend Interface and Implementations for LLM Long-Term Memory
Provides pluggable storage backends: SQLite, PostgreSQL, MongoDB, Redis
"""

import json
import sqlite3
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Memory:
    """Represents a single memory entry"""
    id: str
    content: str
    category: str
    importance: float
    timestamp: str
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert memory to dictionary"""
        return asdict(self)


class StorageBackend(ABC):
    """Abstract base class for storage backends"""
    
    @abstractmethod
    def init_storage(self) -> None:
        """Initialize the storage (create tables, collections, etc.)"""
        pass
    
    @abstractmethod
    def save_memory(self, memory: Memory) -> None:
        """Save a memory to storage"""
        pass
    
    @abstractmethod
    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID"""
        pass
    
    @abstractmethod
    def get_all_memories(self) -> List[Memory]:
        """Retrieve all memories"""
        pass
    
    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID"""
        pass
    
    @abstractmethod
    def search_memories(self, query: str, category: Optional[str] = None) -> List[Memory]:
        """Search memories by content"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the storage connection"""
        pass


class SQLiteBackend(StorageBackend):
    """SQLite storage backend (default, no external dependencies)"""
    
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self.init_storage()
    
    def init_storage(self) -> None:
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    category TEXT,
                    importance REAL,
                    timestamp TEXT,
                    embedding TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON memories(category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)")
            conn.commit()
    
    def save_memory(self, memory: Memory) -> None:
        """Save memory to SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO memories 
                (id, content, category, importance, timestamp, embedding, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.id,
                memory.content,
                memory.category,
                memory.importance,
                memory.timestamp,
                json.dumps(memory.embedding) if memory.embedding else None,
                json.dumps(memory.metadata) if memory.metadata else None
            ))
            conn.commit()
    
    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get memory by ID from SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
            row = cursor.fetchone()
            return self._row_to_memory(row) if row else None
    
    def get_all_memories(self) -> List[Memory]:
        """Get all memories from SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM memories ORDER BY timestamp DESC")
            return [self._row_to_memory(row) for row in cursor.fetchall()]
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete memory from SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def search_memories(self, query: str, category: Optional[str] = None) -> List[Memory]:
        """Search memories in SQLite"""
        sql = "SELECT * FROM memories WHERE content LIKE ?"
        params = [f"%{query}%"]
        
        if category:
            sql += " AND category = ?"
            params.append(category)
        
        sql += " ORDER BY importance DESC, timestamp DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(sql, params)
            return [self._row_to_memory(row) for row in cursor.fetchall()]
    
    def close(self) -> None:
        """SQLite connections are managed per-operation, no persistent connection to close"""
        pass
    
    def _row_to_memory(self, row) -> Memory:
        """Convert SQLite row to Memory object"""
        return Memory(
            id=row[0],
            content=row[1],
            category=row[2],
            importance=row[3],
            timestamp=row[4],
            embedding=json.loads(row[5]) if row[5] else None,
            metadata=json.loads(row[6]) if row[6] else None
        )


class PostgreSQLBackend(StorageBackend):
    """PostgreSQL storage backend for production use"""
    
    def __init__(self, connection_string: str):
        """
        Initialize PostgreSQL backend.
        
        Args:
            connection_string: PostgreSQL connection string
                e.g., "postgresql://user:password@localhost:5432/memory_db"
        """
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            self.psycopg2 = psycopg2
            self.RealDictCursor = RealDictCursor
        except ImportError:
            raise ImportError("psycopg2 is required for PostgreSQL backend. Install with: pip install psycopg2-binary")
        
        self.connection_string = connection_string
        self.conn = None
        self.init_storage()
    
    def _get_connection(self):
        """Get or create database connection"""
        if self.conn is None or self.conn.closed:
            self.conn = self.psycopg2.connect(self.connection_string)
        return self.conn
    
    def init_storage(self) -> None:
        """Initialize PostgreSQL tables"""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    category TEXT,
                    importance REAL,
                    timestamp TEXT,
                    embedding JSONB,
                    metadata JSONB
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_content ON memories USING gin(to_tsvector('english', content))")
            conn.commit()
    
    def save_memory(self, memory: Memory) -> None:
        """Save memory to PostgreSQL"""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO memories (id, content, category, importance, timestamp, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    category = EXCLUDED.category,
                    importance = EXCLUDED.importance,
                    timestamp = EXCLUDED.timestamp,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
            """, (
                memory.id,
                memory.content,
                memory.category,
                memory.importance,
                memory.timestamp,
                json.dumps(memory.embedding) if memory.embedding else None,
                json.dumps(memory.metadata) if memory.metadata else None
            ))
            conn.commit()
    
    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get memory by ID from PostgreSQL"""
        conn = self._get_connection()
        with conn.cursor(cursor_factory=self.RealDictCursor) as cur:
            cur.execute("SELECT * FROM memories WHERE id = %s", (memory_id,))
            row = cur.fetchone()
            return self._row_to_memory(row) if row else None
    
    def get_all_memories(self) -> List[Memory]:
        """Get all memories from PostgreSQL"""
        conn = self._get_connection()
        with conn.cursor(cursor_factory=self.RealDictCursor) as cur:
            cur.execute("SELECT * FROM memories ORDER BY timestamp DESC")
            return [self._row_to_memory(row) for row in cur.fetchall()]
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete memory from PostgreSQL"""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memories WHERE id = %s", (memory_id,))
            conn.commit()
            return cur.rowcount > 0
    
    def search_memories(self, query: str, category: Optional[str] = None) -> List[Memory]:
        """Full-text search in PostgreSQL"""
        conn = self._get_connection()
        with conn.cursor(cursor_factory=self.RealDictCursor) as cur:
            if category:
                cur.execute("""
                    SELECT * FROM memories 
                    WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                    AND category = %s
                    ORDER BY importance DESC, timestamp DESC
                """, (query, category))
            else:
                cur.execute("""
                    SELECT * FROM memories 
                    WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                    ORDER BY importance DESC, timestamp DESC
                """, (query,))
            return [self._row_to_memory(row) for row in cur.fetchall()]
    
    def close(self) -> None:
        """Close PostgreSQL connection"""
        if self.conn and not self.conn.closed:
            self.conn.close()
    
    def _row_to_memory(self, row: Dict) -> Memory:
        """Convert PostgreSQL row to Memory object"""
        return Memory(
            id=row['id'],
            content=row['content'],
            category=row['category'],
            importance=row['importance'],
            timestamp=row['timestamp'],
            embedding=row['embedding'] if row['embedding'] else None,
            metadata=row['metadata'] if row['metadata'] else None
        )


class MongoDBBackend(StorageBackend):
    """MongoDB storage backend for document-based storage"""
    
    def __init__(self, connection_string: str, database: str = "memory_db", collection: str = "memories"):
        """
        Initialize MongoDB backend.
        
        Args:
            connection_string: MongoDB connection string
                e.g., "mongodb://localhost:27017" or "mongodb+srv://..."
            database: Database name
            collection: Collection name
        """
        try:
            from pymongo import MongoClient
            self.MongoClient = MongoClient
        except ImportError:
            raise ImportError("pymongo is required for MongoDB backend. Install with: pip install pymongo")
        
        self.connection_string = connection_string
        self.database_name = database
        self.collection_name = collection
        self.client = None
        self.init_storage()
    
    def _get_collection(self):
        """Get MongoDB collection"""
        if self.client is None:
            self.client = self.MongoClient(self.connection_string)
        return self.client[self.database_name][self.collection_name]
    
    def init_storage(self) -> None:
        """Initialize MongoDB indexes"""
        collection = self._get_collection()
        collection.create_index("id", unique=True)
        collection.create_index("category")
        collection.create_index("importance")
        collection.create_index([("content", "text")])
    
    def save_memory(self, memory: Memory) -> None:
        """Save memory to MongoDB"""
        collection = self._get_collection()
        doc = memory.to_dict()
        collection.update_one({"id": memory.id}, {"$set": doc}, upsert=True)
    
    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get memory by ID from MongoDB"""
        collection = self._get_collection()
        doc = collection.find_one({"id": memory_id})
        return self._doc_to_memory(doc) if doc else None
    
    def get_all_memories(self) -> List[Memory]:
        """Get all memories from MongoDB"""
        collection = self._get_collection()
        docs = collection.find().sort("timestamp", -1)
        return [self._doc_to_memory(doc) for doc in docs]
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete memory from MongoDB"""
        collection = self._get_collection()
        result = collection.delete_one({"id": memory_id})
        return result.deleted_count > 0
    
    def search_memories(self, query: str, category: Optional[str] = None) -> List[Memory]:
        """Full-text search in MongoDB"""
        collection = self._get_collection()
        filter_query = {"$text": {"$search": query}}
        if category:
            filter_query["category"] = category
        
        docs = collection.find(filter_query).sort([("importance", -1), ("timestamp", -1)])
        return [self._doc_to_memory(doc) for doc in docs]
    
    def close(self) -> None:
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
    
    def _doc_to_memory(self, doc: Dict) -> Memory:
        """Convert MongoDB document to Memory object"""
        return Memory(
            id=doc['id'],
            content=doc['content'],
            category=doc.get('category'),
            importance=doc.get('importance', 0.5),
            timestamp=doc.get('timestamp'),
            embedding=doc.get('embedding'),
            metadata=doc.get('metadata')
        )


class RedisBackend(StorageBackend):
    """Redis storage backend for high-performance caching"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, password: Optional[str] = None):
        """
        Initialize Redis backend.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (optional)
        """
        try:
            import redis
            self.redis = redis
        except ImportError:
            raise ImportError("redis is required for Redis backend. Install with: pip install redis")
        
        self.client = self.redis.Redis(host=host, port=port, db=db, password=password, decode_responses=True)
        self.prefix = "memory:"
        self.init_storage()
    
    def init_storage(self) -> None:
        """Redis doesn't require schema initialization"""
        pass
    
    def save_memory(self, memory: Memory) -> None:
        """Save memory to Redis"""
        key = f"{self.prefix}{memory.id}"
        self.client.hset(key, mapping={
            "id": memory.id,
            "content": memory.content,
            "category": memory.category or "",
            "importance": str(memory.importance),
            "timestamp": memory.timestamp or "",
            "embedding": json.dumps(memory.embedding) if memory.embedding else "",
            "metadata": json.dumps(memory.metadata) if memory.metadata else ""
        })
        # Add to sorted set for ordering
        self.client.zadd("memories:by_timestamp", {memory.id: float(hash(memory.timestamp or ""))})
        if memory.category:
            self.client.sadd(f"memories:category:{memory.category}", memory.id)
    
    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get memory by ID from Redis"""
        key = f"{self.prefix}{memory_id}"
        data = self.client.hgetall(key)
        return self._data_to_memory(data) if data else None
    
    def get_all_memories(self) -> List[Memory]:
        """Get all memories from Redis"""
        keys = self.client.keys(f"{self.prefix}*")
        memories = []
        for key in keys:
            data = self.client.hgetall(key)
            if data:
                memories.append(self._data_to_memory(data))
        return sorted(memories, key=lambda m: m.timestamp or "", reverse=True)
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete memory from Redis"""
        key = f"{self.prefix}{memory_id}"
        memory = self.get_memory(memory_id)
        if memory:
            self.client.delete(key)
            self.client.zrem("memories:by_timestamp", memory_id)
            if memory.category:
                self.client.srem(f"memories:category:{memory.category}", memory_id)
            return True
        return False
    
    def search_memories(self, query: str, category: Optional[str] = None) -> List[Memory]:
        """Search memories in Redis (basic implementation)"""
        # Redis doesn't have built-in full-text search, so we do a simple scan
        # For production, consider RediSearch module
        all_memories = self.get_all_memories()
        query_lower = query.lower()
        
        results = [m for m in all_memories if query_lower in m.content.lower()]
        if category:
            results = [m for m in results if m.category == category]
        
        return sorted(results, key=lambda m: m.importance, reverse=True)
    
    def close(self) -> None:
        """Close Redis connection"""
        self.client.close()
    
    def _data_to_memory(self, data: Dict) -> Memory:
        """Convert Redis hash to Memory object"""
        return Memory(
            id=data['id'],
            content=data['content'],
            category=data['category'] if data['category'] else None,
            importance=float(data['importance']) if data['importance'] else 0.5,
            timestamp=data['timestamp'] if data['timestamp'] else None,
            embedding=json.loads(data['embedding']) if data['embedding'] else None,
            metadata=json.loads(data['metadata']) if data['metadata'] else None
        )


def create_storage_backend(backend_type: str = "sqlite", **kwargs) -> StorageBackend:
    """
    Factory function to create storage backends.
    
    Args:
        backend_type: One of "sqlite", "postgresql", "mongodb", "redis"
        **kwargs: Backend-specific configuration
        
    Returns:
        StorageBackend instance
        
    Examples:
        # SQLite (default)
        backend = create_storage_backend("sqlite", db_path="memory.db")
        
        # PostgreSQL
        backend = create_storage_backend("postgresql", 
            connection_string="postgresql://user:pass@localhost:5432/memory_db")
        
        # MongoDB
        backend = create_storage_backend("mongodb",
            connection_string="mongodb://localhost:27017",
            database="memory_db",
            collection="memories")
        
        # Redis
        backend = create_storage_backend("redis",
            host="localhost",
            port=6379,
            password="optional_password")
    """
    backends = {
        "sqlite": SQLiteBackend,
        "postgresql": PostgreSQLBackend,
        "postgres": PostgreSQLBackend,
        "mongodb": MongoDBBackend,
        "mongo": MongoDBBackend,
        "redis": RedisBackend,
    }
    
    backend_type = backend_type.lower()
    if backend_type not in backends:
        raise ValueError(f"Unknown backend type: {backend_type}. Supported: {list(backends.keys())}")
    
    return backends[backend_type](**kwargs)
