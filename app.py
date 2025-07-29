"""
Streamlit Test Interface for Long-Term Memory System
Provides a comprehensive UI to test all memory system capabilities.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import tempfile
import uuid

from memory_system import LongTermMemorySystem, Memory

# Page configuration
st.set_page_config(
    page_title="Long-Term Memory System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .memory-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .memory-content {
        font-weight: bold;
        color: #1f77b4;
    }
    .memory-meta {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #f5c6cb;
    }
    .setup-container {
        max-width: 600px;
        margin: 2rem auto;
        padding: 2rem;
        background-color: #f8f9fa;
        border-radius: 1rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

def initialize_memory_system(api_key):
    """Initialize the memory system with the provided API key and session-specific database"""
    # Create a unique database file for this session
    if "session_db_path" not in st.session_state:
        temp_dir = tempfile.gettempdir()
        session_id = str(uuid.uuid4())[:8]
        st.session_state.session_db_path = os.path.join(temp_dir, f"memory_session_{session_id}.db")
        st.session_state.session_id = session_id
    
    return LongTermMemorySystem(api_key, st.session_state.session_db_path)

def cleanup_session_database():
    """Clean up the session database file"""
    if "session_db_path" in st.session_state:
        try:
            if os.path.exists(st.session_state.session_db_path):
                os.remove(st.session_state.session_db_path)
        except Exception as e:
            pass

def reset_session():
    """Reset the current session and create a new one"""
    cleanup_session_database()
    
    keys_to_clear = ["session_db_path", "session_id", "memory_system", "chat_history"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

def check_api_key():
    """Check if API key is available and valid"""
    if "openai_api_key" not in st.session_state:
        return False
    
    api_key = st.session_state.openai_api_key
    if not api_key or not api_key.startswith("sk-"):
        return False
    
    return True

def setup_page():
    """Display the setup page to get OpenAI API key from user"""
    st.markdown("""
    <div class="setup-container">
        <h1 style="text-align: center; color: #1f77b4;">🧠 Long-Term Memory System</h1>
        <h3 style="text-align: center; color: #666;">Setup Required</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔑 OpenAI API Key Required")
    st.markdown("""
    To use the Long-Term Memory System, you need to provide your OpenAI API key. 
    This key will only be stored for this session and will not be saved permanently.
    
    **🗃️ Session-Based Memory:** Each session creates a fresh database. Your memories will only persist during this browser session.
    
    **🔍 Enhanced Search:** Uses OpenAI's text-embedding-3-small model for superior semantic search capabilities.
    """)
    
    st.markdown("---")
    api_key_input = st.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        placeholder="sk-...",
        help="Your API key will only be stored for this session"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Start Using Memory System", type="primary", use_container_width=True):
            if api_key_input.strip():
                if api_key_input.startswith("sk-"):
                    st.session_state.openai_api_key = api_key_input.strip()
                    st.success("✅ API key saved! Initializing memory system...")
                    st.rerun()
                else:
                    st.error("❌ Invalid API key format. OpenAI API keys start with 'sk-'")
            else:
                st.error("❌ Please enter your OpenAI API key")
    
    with col2:
        if st.button("ℹ️ How to get API Key", use_container_width=True):
            st.markdown("""
            ### How to get your OpenAI API Key:
            
            1. Go to [OpenAI Platform](https://platform.openai.com/)
            2. Sign in to your account
            3. Navigate to **API Keys** section
            4. Click **Create new secret key**
            5. Copy the key and paste it above
            
            **Note:** Make sure you have credits in your OpenAI account to use the API.
            """)
    
    st.markdown("---")
    
    with st.expander("🔒 Privacy & Security Information"):
        st.markdown("""
        **Your API Key Security:**
        - Your API key is only stored in this browser session
        - It's not saved to any file or database
        - It will be cleared when you close the browser tab
        - The key is only used to communicate with OpenAI's API
        
        **Session-Based Memory Storage:**
        - Each session creates a temporary database file
        - Memories are stored locally during your session
        - Database is automatically cleaned up when you start a new session
        - No data persists between different browser sessions
        
        **What this app does:**
        - Extracts memories from your conversations using OpenAI GPT
        - Stores memories temporarily in a session-specific SQLite database
        - Uses OpenAI's text-embedding-3-small model for semantic search
        - Uses OpenAI's API for memory extraction and question answering
        - Provides superior semantic search capabilities with OpenAI embeddings
        """)
    
    with st.expander("🚀 Features Overview"):
        st.markdown("""
        **Long-Term Memory System Features:**
        - **💬 AI Chat:** Conversational AI that remembers everything about you
        - **🧠 Memory Creation:** Extract and store memories from your messages
        - **🔍 Memory Search:** Find relevant memories using semantic search
        - **📊 Analytics:** Visualize your memory patterns and statistics
        - **⚙️ Management:** View, filter, and manage your stored memories
        """)
    
    if "session_id" in st.session_state:
        st.markdown("---")
        st.info(f"🔄 Current Session ID: {st.session_state.session_id}")
        if st.button("🗑️ Start New Session", help="This will clear all current memories and start fresh"):
            reset_session()
            st.rerun()

def cleanup_session_database():
    """Clean up the session database file"""
    if "session_db_path" in st.session_state:
        try:
            if os.path.exists(st.session_state.session_db_path):
                os.remove(st.session_state.session_db_path)
                st.info(f"Cleaned up session database: {st.session_state.session_db_path}")
        except Exception as e:
            st.warning(f"Failed to cleanup session database: {e}")

def reset_session():
    """Reset the current session and create a new one"""
    cleanup_session_database()
    
    keys_to_clear = ["session_db_path", "session_id", "memory_system", "chat_history"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

def display_memory_card(memory: Memory):
    """Display a memory in a card format"""
    st.markdown(f"""
    <div class="memory-card">
        <div class="memory-content">{memory.content}</div>
        <div class="memory-meta">
            <strong>Category:</strong> {memory.category} | 
            <strong>Importance:</strong> {memory.importance:.2f} | 
            <strong>ID:</strong> {memory.id[:20]}... | 
            <strong>Time:</strong> {memory.timestamp[:19]}
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Check if API key is available
    if not check_api_key():
        setup_page()
        return
    
    st.title("🧠 Long-Term Memory System Interface")
    st.markdown("Test and interact with the advanced long-term memory system for LLM agents.")
    
    # Initialize memory system with session API key
    try:
        if "memory_system" not in st.session_state:
            with st.spinner("Initializing memory system..."):
                st.session_state.memory_system = initialize_memory_system(st.session_state.openai_api_key)
        
        memory_system = st.session_state.memory_system
        
    except Exception as e:
        st.error(f"Failed to initialize memory system: {e}")
        st.info("Please check your API key and try again.")
        
        # Add button to reset API key
        if st.button("🔄 Reset API Key"):
            if "openai_api_key" in st.session_state:
                del st.session_state.openai_api_key
            if "memory_system" in st.session_state:
                del st.session_state.memory_system
            st.rerun()
        return
    
    # Sidebar for navigation and API key management
    st.sidebar.title("Navigation")
    
    # API key status in sidebar
    with st.sidebar:
        st.markdown("### 🔑 API Key Status")
        st.success("✅ API Key Active")
        
        # Show masked API key
        masked_key = st.session_state.openai_api_key[:7] + "..." + st.session_state.openai_api_key[-4:]
        st.caption(f"Key: {masked_key}")
        
        # Session information
        st.markdown("### 📊 Session Info")
        if "session_id" in st.session_state:
            st.caption(f"Session ID: {st.session_state.session_id}")
        st.caption("🗃️ Fresh database for this session")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Change API Key", use_container_width=True):
                if "openai_api_key" in st.session_state:
                    del st.session_state.openai_api_key
                if "memory_system" in st.session_state:
                    del st.session_state.memory_system
                st.rerun()
        
        with col2:
            if st.button("🗑️ New Session", use_container_width=True, help="Start fresh with empty memory"):
                reset_session()
                st.rerun()
        
        st.markdown("---")
    
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["💬 Chat", "🧠 Memory Creation", "🔍 Query Memories", "📊 Memory Analytics", "⚙️ Memory Management"]
    )
    
    if page == "💬 Chat":
        chat_page(memory_system)
    elif page == "🧠 Memory Creation":
        memory_creation_page(memory_system)
    elif page == "🔍 Query Memories":
        query_memories_page(memory_system)
    elif page == "📊 Memory Analytics":
        analytics_page(memory_system)
    elif page == "⚙️ Memory Management":
        management_page(memory_system)

def chat_page(memory_system):
    """Page for chatbot-style conversation with AI using memory context"""
    st.header("💬 AI Chatbot")
    st.markdown("Chat with your AI assistant that remembers everything about you!")
    
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        # Add welcome message
        st.session_state.chat_history.append(("assistant", "Hello! I'm your AI assistant with long-term memory. I can remember our conversations and help you with anything. What would you like to talk about?"))
    
    # Display chat messages using Streamlit's chat message component
    st.subheader("Conversation")
    
    # Create a container for the chat with custom height
    chat_container = st.container()
    
    with chat_container:
        # Display all messages
        for role, message in st.session_state.chat_history:
            if role == "user":
                with st.chat_message("user", avatar="👤"):
                    st.write(message)
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(message)
    
    # Input area at the bottom
    st.markdown("---")
    
    # Use Streamlit's chat input component
    user_input = st.chat_input("Type your message here...")
    
    # Handle user input
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append(("user", user_input))
        
        # Display the user message immediately
        with st.chat_message("user", avatar="👤"):
            st.write(user_input)
        
        # Show AI response with spinner
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    # Process the message for memory extraction (silently)
                    memory_system.process_message(user_input, "default_user", "")
                    
                    # Get AI response using memory
                    ai_response = memory_system.answer_with_memory(user_input)
                    
                    # Display the response
                    st.write(ai_response)
                    
                    # Add AI response to history
                    st.session_state.chat_history.append(("assistant", ai_response))
                    
                except Exception as e:
                    error_message = f"Sorry, I encountered an error: {str(e)}"
                    st.write(error_message)
                    st.session_state.chat_history.append(("assistant", error_message))
                    
                    # If it's an API key related error, suggest resetting
                    if "api" in str(e).lower() or "auth" in str(e).lower():
                        st.error("This might be an API key issue. Try resetting your API key in the sidebar.")
    
    # Chat controls in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.chat_history.append(("assistant", "Hello! I'm your AI assistant with long-term memory. I can remember our conversations and help you with anything. What would you like to talk about?"))
            st.rerun()
    
    with col2:
        if st.button("🔗 Show Relevant Memories", use_container_width=True):
            if st.session_state.chat_history:
                last_user_message = None
                for role, message in reversed(st.session_state.chat_history):
                    if role == "user":
                        last_user_message = message
                        break
                
                if last_user_message:
                    with st.expander("🧠 Relevant Memories", expanded=True):
                        try:
                            memories = memory_system.query_memories(last_user_message, 5)
                            if memories:
                                for memory in memories:
                                    st.markdown(f"**{memory.content}**")
                                    st.caption(f"Category: {memory.category} | Importance: {memory.importance:.2f}")
                                    st.markdown("---")
                            else:
                                st.info("No relevant memories found.")
                        except Exception as e:
                            st.error(f"Error retrieving memories: {e}")
    
    with col3:
        if st.button("💾 Export Chat", use_container_width=True):
            chat_export = []
            for role, message in st.session_state.chat_history:
                chat_export.append(f"{role.title()}: {message}")
            
            export_text = "\n\n".join(chat_export)
            st.download_button(
                label="📥 Download",
                data=export_text,
                file_name="chat_history.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    # Sidebar with chat info and quick actions
    with st.sidebar:
        st.markdown("### 💬 Chat Statistics")
        
        # Chat statistics
        total_messages = len(st.session_state.chat_history)
        user_messages = len([msg for role, msg in st.session_state.chat_history if role == "user"])
        
        st.metric("Total Messages", total_messages)
        st.metric("Your Messages", user_messages)
        
        # Memory stats
        try:
            stats = memory_system.get_memory_stats()
            st.metric("Available Memories", stats["total_memories"])
            
            if stats["categories"]:
                st.markdown("### 📂 Memory Categories")
                for category, count in stats["categories"].items():
                    st.write(f"**{category}:** {count}")
        except:
            st.metric("Available Memories", "Error")
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🧠 Manage Memories", use_container_width=True):
            st.switch_page("🧠 Memory Creation")
        
        if st.button("🔍 Search Memories", use_container_width=True):
            st.switch_page("🔍 Query Memories")
        
        if st.button("📊 View Analytics", use_container_width=True):
            st.switch_page("📊 Memory Analytics")

def memory_creation_page(memory_system):
    """Page for creating and managing memories"""
    st.header("🧠 Memory Creation")
    st.markdown("Extract and store memories from your messages.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Create New Memories")
        
        # User input
        user_message = st.text_area(
            "Enter your message:",
            placeholder="e.g., 'I use Shram and Magnet as productivity tools'",
            height=100
        )
        
        user_id = st.text_input("User ID (optional):", value="default_user")
        context = st.text_area("Additional context (optional):", height=60)
        
        if st.button("Extract Memories", type="primary"):
            if user_message.strip():
                with st.spinner("Processing message and extracting memories..."):
                    try:
                        result = memory_system.process_message(user_message, user_id, context)
                        
                        # Display results
                        if result["new_memories"]:
                            st.markdown('<div class="success-message">✅ New memories created!</div>', unsafe_allow_html=True)
                            for mem in result["new_memories"]:
                                st.markdown(f"**Memory:** {mem['content']}")
                                st.markdown(f"**Category:** {mem['category']} | **Importance:** {mem['importance']}")
                                st.markdown("---")
                        
                        if result["updated_memories"]:
                            st.markdown('<div class="success-message">🔄 Memories updated!</div>', unsafe_allow_html=True)
                            for update in result["updated_memories"]:
                                st.markdown(f"**Updated:** {update['reason']}")
                        
                        if result["deleted_memories"]:
                            st.markdown('<div class="success-message">🗑️ Memories deleted!</div>', unsafe_allow_html=True)
                            for deletion in result["deleted_memories"]:
                                st.markdown(f"**Deleted:** {deletion['reason']}")
                        
                        if not any([result["new_memories"], result["updated_memories"], result["deleted_memories"]]):
                            st.info("No memories were extracted from this message.")
                            
                    except Exception as e:
                        st.markdown(f'<div class="error-message">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
                        if "api" in str(e).lower() or "auth" in str(e).lower():
                            st.error("This might be an API key issue. Try resetting your API key in the sidebar.")
            else:
                st.warning("Please enter a message.")
        
        # Recent memories
        st.subheader("Recently Created Memories")
        try:
            memories = memory_system.get_all_memories()
            recent_memories = sorted(memories, key=lambda x: x.timestamp, reverse=True)[:5]
            
            if recent_memories:
                for memory in recent_memories:
                    display_memory_card(memory)
            else:
                st.info("No memories created yet.")
        except Exception as e:
            st.error(f"Error loading recent memories: {e}")
    
    with col2:
        st.subheader("Memory Statistics")
        try:
            stats = memory_system.get_memory_stats()
            st.metric("Total Memories", stats["total_memories"])
            st.metric("Avg Importance", f"{stats['avg_importance']:.2f}")
            
            if stats["categories"]:
                st.subheader("Categories")
                for category, count in stats["categories"].items():
                    st.write(f"**{category}:** {count}")
        except Exception as e:
            st.error(f"Error loading stats: {e}")
        
        # Quick actions
        st.subheader("Quick Actions")
        
        if st.button("View All Memories"):
            st.switch_page("Memory Management")
        
        if st.button("Test Memory Query"):
            st.switch_page("Query Memories")

def query_memories_page(memory_system):
    """Page for querying memories"""
    st.header("🔍 Query Memories")
    
    st.subheader("Memory Search")
    query = st.text_input(
        "Enter your query:",
        placeholder="e.g., 'What productivity tools do I use?'"
    )
    
    max_results = st.slider("Max results:", 1, 20, 5)
    
    if st.button("Search Memories", type="primary"):
        if query.strip():
            with st.spinner("Searching memories..."):
                try:
                    memories = memory_system.query_memories(query, max_results)
                    
                    if memories:
                        st.success(f"Found {len(memories)} relevant memories:")
                        for memory in memories:
                            display_memory_card(memory)
                    else:
                        st.info("No relevant memories found.")
                except Exception as e:
                    st.error(f"Error searching memories: {e}")
                    if "api" in str(e).lower() or "auth" in str(e).lower():
                        st.error("This might be an API key issue. Try resetting your API key in the sidebar.")
        else:
            st.warning("Please enter a query.")

def analytics_page(memory_system):
    """Page for memory analytics and visualization"""
    st.header("📊 Memory Analytics")
    
    try:
        memories = memory_system.get_all_memories()
        
        if not memories:
            st.info("No memories found. Create some memories first!")
            return
        
        # Convert to DataFrame for analysis
        df_data = []
        for memory in memories:
            df_data.append({
                'id': memory.id,
                'content': memory.content,
                'category': memory.category,
                'importance': memory.importance,
                'timestamp': memory.timestamp,
                'content_length': len(memory.content)
            })
        
        df = pd.DataFrame(df_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Memories", len(memories))
        with col2:
            st.metric("Categories", df['category'].nunique())
        with col3:
            st.metric("Avg Importance", f"{df['importance'].mean():.2f}")
        with col4:
            st.metric("Latest Memory", df['timestamp'].max().strftime("%Y-%m-%d"))
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution
            st.subheader("Memory Distribution by Category")
            category_counts = df['category'].value_counts()
            fig_pie = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Memory Categories"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Importance distribution
            st.subheader("Importance Distribution")
            fig_hist = px.histogram(
                df,
                x='importance',
                nbins=20,
                title="Memory Importance Scores"
            )
            st.plotly_chart(fig_hist, use_container_width=True)    
        # Memory details table
        st.subheader("Memory Details")
        display_df = df[['content', 'category', 'importance', 'timestamp']].copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(display_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading analytics: {e}")

def management_page(memory_system):
    """Page for memory management operations"""
    st.header("⚙️ Memory Management")
    
    tab1, tab2, tab3 = st.tabs(["View All Memories", "Delete Memories", "System Info"])
    
    with tab1:
        st.subheader("All Stored Memories")
        try:
            memories = memory_system.get_all_memories()
            
            if memories:
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    categories = list(set(m.category for m in memories))
                    selected_category = st.selectbox("Filter by category:", ["All"] + categories)
                
                with col2:
                    min_importance = st.slider("Minimum importance:", 0.0, 1.0, 0.0)
                
                # Apply filters
                filtered_memories = memories
                if selected_category != "All":
                    filtered_memories = [m for m in filtered_memories if m.category == selected_category]
                filtered_memories = [m for m in filtered_memories if m.importance >= min_importance]
                
                st.write(f"Showing {len(filtered_memories)} of {len(memories)} memories")
                
                for memory in filtered_memories:
                    display_memory_card(memory)
            else:
                st.info("No memories found.")
        except Exception as e:
            st.error(f"Error loading memories: {e}")
    
    with tab2:
        st.subheader("Delete Memories")
        st.warning("⚠️ Memory deletion is permanent!")
        
        try:
            memories = memory_system.get_all_memories()
            
            if memories:
                # Create a selectbox with memory previews
                memory_options = {}
                for memory in memories:
                    preview = f"{memory.content[:50]}..." if len(memory.content) > 50 else memory.content
                    memory_options[f"{preview} (ID: {memory.id[:8]}...)"] = memory.id
                
                selected_memory = st.selectbox(
                    "Select memory to delete:",
                    [""] + list(memory_options.keys())
                )
                
                if selected_memory:
                    memory_id = memory_options[selected_memory]
                    memory = memory_system.database.get_memory(memory_id)
                    
                    if memory:
                        st.write("**Memory to delete:**")
                        display_memory_card(memory)
                        
                        if st.button("🗑️ Delete Memory", type="secondary"):
                            if memory_system.delete_memory(memory_id):
                                st.success("Memory deleted successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to delete memory.")
            else:
                st.info("No memories to delete.")
        except Exception as e:
            st.error(f"Error in delete operation: {e}")
    
    with tab3:
        st.subheader("System Information")
        
        try:
            stats = memory_system.get_memory_stats()
            
            st.json(stats)
            
            # Database info
            st.subheader("Database Info")
            if "session_db_path" in st.session_state:
                st.write(f"**Session Database:** {os.path.basename(st.session_state.session_db_path)}")
                st.write(f"**Session ID:** {st.session_state.get('session_id', 'Unknown')}")
                st.caption("🗃️ This database is temporary and will be cleaned up when you start a new session")
            else:
                st.write(f"**Database Path:** {memory_system.database.db_path}")
            
            # Vector store info
            st.subheader("Vector Store Info")
            st.write(f"**Embedding Model:** OpenAI text-embedding-3-small")
            st.write(f"**Embedding Dimension:** {memory_system.vector_store.dimension}")
            st.write(f"**Total Vectors:** {memory_system.vector_store.index.ntotal}")
            st.caption("🚀 Using OpenAI's advanced embedding model for superior semantic search")
            
            # Session controls
            st.subheader("Session Controls")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🗑️ Clear All Memories", type="secondary", use_container_width=True):
                    if st.button("⚠️ Confirm Clear All", type="secondary", use_container_width=True):
                        reset_session()
                        st.success("All memories cleared! Starting fresh session...")
                        st.rerun()
            
            with col2:
                if st.button("🔄 Start New Session", use_container_width=True):
                    reset_session()
                    st.success("New session started!")
                    st.rerun()
            
        except Exception as e:
            st.error(f"Error loading system info: {e}")

if __name__ == "__main__":
    main()
