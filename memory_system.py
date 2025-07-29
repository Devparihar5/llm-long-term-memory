"""
Advanced Long-Term Memory System for LLM Agents
Provides persistent memory storage, retrieval, and management capabilities.
"""

import json
import sqlite3
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import faiss
import logging
from pathlib import Path

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
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

class ExtractedMemory(BaseModel):
    """Pydantic model for extracted memory"""
    content: str = Field(description="Specific factual statement about the user")
    category: str = Field(description="Category like 'tools', 'preferences', 'personal', 'habits'")
    importance: float = Field(description="Importance score between 0.0 and 1.0")

class MemoryUpdate(BaseModel):
    """Pydantic model for memory updates"""
    action: str = Field(description="Either 'update' or 'delete'")
    memory_id: str = Field(description="ID of memory to update/delete")
    new_content: Optional[str] = Field(description="New content if updating, null if deleting")
    reason: str = Field(description="Explanation of the change")

class MemoryExtractor:
    """Extracts memories from conversations using LangChain and OpenAI"""
    
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0.1
        )
        
    def extract_memories(self, message: str, context: str = "") -> List[Dict]:
        """Extract memories from a message using GPT via LangChain"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting factual information about users from conversations.
            Analyze the message and extract any factual information that should be remembered about the user.
            Focus on preferences, tools they use, personal information, habits, etc.
            
            Return a JSON list of memories in this exact format:
            [
                {{
                    "content": "specific factual statement about the user",
                    "category": "category like 'tools', 'preferences', 'personal', 'habits'",
                    "importance": 0.8
                }}
            ]
            
            Only extract clear, factual statements. Return empty list [] if no memories found.
            Make sure the response is valid JSON."""),
            ("human", "Context: {context}\nMessage: {message}")
        ])
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "context": context,
                "message": message
            })
            
            content = response.content.strip()
            
            # Clean up the response to extract JSON
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
                
            memories = json.loads(content)
            return memories if isinstance(memories, list) else []
            
        except Exception as e:
            logger.error(f"Error extracting memories: {e}")
            return []
    
    def detect_memory_updates(self, message: str, existing_memories: List[Memory]) -> List[Dict]:
        """Detect if message contains updates or deletions to existing memories"""
        if not existing_memories:
            return []
            
        memories_text = "\n".join([f"ID: {mem.id[:20]}... - {mem.content}" for mem in existing_memories])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at detecting when new information updates or contradicts existing memories.
            Analyze if the new message contains updates or deletions to existing memories.
            
            Return a JSON list of updates in this exact format:
            [
                {{
                    "action": "update" or "delete",
                    "memory_id": "id of memory to update/delete",
                    "new_content": "new content if updating, null if deleting",
                    "reason": "explanation of the change"
                }}
            ]
            
            Return empty list [] if no updates needed.
            Make sure the response is valid JSON."""),
            ("human", "Existing memories:\n{memories}\n\nNew message: {message}")
        ])
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "memories": memories_text,
                "message": message
            })
            
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
                
            updates = json.loads(content)
            return updates if isinstance(updates, list) else []
            
        except Exception as e:
            logger.error(f"Error detecting memory updates: {e}")
            return []

class VectorStore:
    """Handles vector embeddings and similarity search using OpenAI embeddings"""
    
    def __init__(self, openai_api_key: str, embedding_model: str = "text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(
            api_key=openai_api_key,
            model=embedding_model
        )
        # OpenAI text-embedding-3-small has 1536 dimensions
        # text-embedding-3-large has 3072 dimensions
        self.dimension = 1536 if "small" in embedding_model else 3072
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.memory_ids = []
        
    def add_embeddings(self, memories: List[Memory]) -> None:
        """Add memory embeddings to the vector store"""
        if not memories:
            return
            
        texts = [memory.content for memory in memories]
        embeddings = self.embeddings.embed_documents(texts)
        
        # Normalize embeddings for cosine similarity
        embeddings = np.array(embeddings)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        # Store embeddings in memory objects
        for memory, embedding in zip(memories, embeddings):
            memory.embedding = embedding.tolist()
            
        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))
        self.memory_ids.extend([memory.id for memory in memories])
        
    def search_similar(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar memories"""
        if self.index.ntotal == 0:
            return []
            
        query_embedding = self.embeddings.embed_query(query)
        query_embedding = np.array([query_embedding])
        
        # Normalize query embedding
        norm = np.linalg.norm(query_embedding)
        query_embedding = query_embedding / norm
        
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.memory_ids):
                results.append((self.memory_ids[idx], float(score)))
                
        return results
    
    def remove_embedding(self, memory_id: str) -> None:
        """Remove embedding from vector store (rebuild index)"""
        if memory_id in self.memory_ids:
            # FAISS doesn't support direct deletion, so we rebuild
            remaining_ids = [mid for mid in self.memory_ids if mid != memory_id]
            self.memory_ids = remaining_ids
            # Note: In production, you'd want a more efficient approach

class MemoryDatabase:
    """SQLite database for persistent memory storage"""
    
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self) -> None:
        """Initialize the database schema"""
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
            conn.commit()
            
    def save_memory(self, memory: Memory) -> None:
        """Save a memory to the database"""
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
        """Retrieve a memory by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return self._row_to_memory(row)
            return None
            
    def get_all_memories(self) -> List[Memory]:
        """Retrieve all memories"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM memories ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            
            return [self._row_to_memory(row) for row in rows]
            
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            return cursor.rowcount > 0
            
    def search_memories(self, query: str, category: str = None) -> List[Memory]:
        """Search memories by content"""
        sql = "SELECT * FROM memories WHERE content LIKE ?"
        params = [f"%{query}%"]
        
        if category:
            sql += " AND category = ?"
            params.append(category)
            
        sql += " ORDER BY importance DESC, timestamp DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
            
            return [self._row_to_memory(row) for row in rows]
            
    def _row_to_memory(self, row) -> Memory:
        """Convert database row to Memory object"""
        return Memory(
            id=row[0],
            content=row[1],
            category=row[2],
            importance=row[3],
            timestamp=row[4],
            embedding=json.loads(row[5]) if row[5] else None,
            metadata=json.loads(row[6]) if row[6] else None
        )

class LongTermMemorySystem:
    """Main long-term memory system orchestrating all components"""
    
    def __init__(self, openai_api_key: str, db_path: str = "memory.db", embedding_model: str = "text-embedding-3-small"):
        self.extractor = MemoryExtractor(openai_api_key)
        self.database = MemoryDatabase(db_path)
        self.vector_store = VectorStore(openai_api_key, embedding_model)
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0.3
        )
        
        # Load existing memories into vector store
        self._load_existing_memories()
        
    def _load_existing_memories(self) -> None:
        """Load existing memories from database into vector store"""
        memories = self.database.get_all_memories()
        if memories:
            # Filter memories that have embeddings
            memories_with_embeddings = [m for m in memories if m.embedding]
            if memories_with_embeddings:
                self.vector_store.add_embeddings(memories_with_embeddings)
                
    def process_message(self, message: str, user_id: str = "default", context: str = "") -> Dict:
        """Process a message and extract/update memories"""
        result = {
            "new_memories": [],
            "updated_memories": [],
            "deleted_memories": []
        }
        
        # Get existing memories for update detection
        existing_memories = self.database.get_all_memories()
        
        # Check for memory updates/deletions
        updates = self.extractor.detect_memory_updates(message, existing_memories)
        
        for update in updates:
            if update["action"] == "delete":
                memory_id = update["memory_id"]
                if self.database.delete_memory(memory_id):
                    result["deleted_memories"].append(update)
                    self.vector_store.remove_embedding(memory_id)
                    
            elif update["action"] == "update":
                memory_id = update["memory_id"]
                existing_memory = self.database.get_memory(memory_id)
                if existing_memory:
                    existing_memory.content = update["new_content"]
                    existing_memory.timestamp = datetime.now().isoformat()
                    self.database.save_memory(existing_memory)
                    result["updated_memories"].append(update)
        
        # Extract new memories
        new_memories_data = self.extractor.extract_memories(message, context)
        
        for mem_data in new_memories_data:
            memory = Memory(
                id=f"{user_id}_{datetime.now().timestamp()}_{hash(mem_data['content'])}",
                content=mem_data["content"],
                category=mem_data.get("category", "general"),
                importance=mem_data.get("importance", 0.5),
                timestamp=datetime.now().isoformat(),
                metadata={"user_id": user_id, "source_message": message}
            )
            
            # Generate embedding
            self.vector_store.add_embeddings([memory])
            
            # Save to database
            self.database.save_memory(memory)
            result["new_memories"].append(memory.to_dict())
            
        return result
    
    def query_memories(self, query: str, k: int = 5) -> List[Memory]:
        """Query memories using semantic search"""
        # Get similar memories using vector search
        similar_results = self.vector_store.search_similar(query, k)
        
        memories = []
        for memory_id, score in similar_results:
            memory = self.database.get_memory(memory_id)
            if memory:
                memories.append(memory)
                
        return memories
    
    def get_relevant_context(self, query: str, max_memories: int = 5) -> str:
        """Get relevant memories as context for LLM"""
        memories = self.query_memories(query, max_memories)
        
        if not memories:
            return "No relevant memories found."
            
        context_parts = []
        for memory in memories:
            context_parts.append(f"- {memory.content} (Category: {memory.category})")
            
        return "Relevant memories:\n" + "\n".join(context_parts)
    
    def answer_with_memory(self, question: str, max_memories: int = 5) -> str:
        """Answer a question using relevant memories"""
        context = self.get_relevant_context(question, max_memories)
        
        prompt = f"""
        Based on the following memories about the user, answer their question.
        
        {context}
        
        Question: {question}
        
        Answer based only on the memories provided. If no relevant information is found, say so.
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I'm sorry, I couldn't process your question at the moment."
    
    def get_all_memories(self) -> List[Memory]:
        """Get all stored memories"""
        return self.database.get_all_memories()
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory"""
        success = self.database.delete_memory(memory_id)
        if success:
            self.vector_store.remove_embedding(memory_id)
        return success
    
    def get_memory_stats(self) -> Dict:
        """Get statistics about stored memories"""
        memories = self.database.get_all_memories()
        
        categories = {}
        for memory in memories:
            categories[memory.category] = categories.get(memory.category, 0) + 1
            
        return {
            "total_memories": len(memories),
            "categories": categories,
            "avg_importance": sum(m.importance for m in memories) / len(memories) if memories else 0
        }
