import streamlit as st
import os
import nest_asyncio
import hashlib
import requests
from datetime import datetime
from urllib.parse import urlparse
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

# Infrastructure and LlamaIndex imports
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_parse import LlamaParse

nest_asyncio.apply()

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="CFO Helper", page_icon="💰", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #000000 !important; }
    h1, h2, h3 { color: #FFB900 !important; }
    p, span, label { color: #ffffff !important; }
    .stChatInput { border-color: #FFB900 !important; background-color: #1a1a1a !important; }
    div[data-testid="stChatMessage"] { background-color: #1a1a1a !important; border: 1px solid #FFB900; border-radius: 10px; }
    [data-testid="stSidebar"] { background-color: #1a1a1a !important; }
    .stButton button { background-color: #FFB900 !important; color: #000000 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- ENVIRONMENT & DATABASE SETUP ---
DATABASE_URL = os.environ.get("NEON_DATABASE_URL", "").strip().replace("\\n", "")
VECTOR_TABLE = "document_vectors"  # LlamaIndex will prefix this with 'data_'

if not DATABASE_URL or not os.environ.get("OPENAI_API_KEY"):
    st.error("Missing environment variables: OPENAI_API_KEY or NEON_DATABASE_URL")
    st.stop()

parsed_url = urlparse(DATABASE_URL)
DB_PARAMS = {
    "dbname": parsed_url.path.lstrip('/'),
    "user": parsed_url.username,
    "password": parsed_url.password,
    "host": parsed_url.hostname,
    "port": str(parsed_url.port or 5432)
}

@st.cache_resource
def get_db_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300,
                         connect_args={"connect_timeout": 10, "sslmode": "require"})

def init_database():
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    file_hash VARCHAR(64) NOT NULL UNIQUE,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
        return True
    except Exception:
        return False

# --- UTILITY FUNCTIONS ---
def get_file_hash(file_content):
    return hashlib.md5(file_content).hexdigest()

def get_documents_list():
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT id, filename, file_hash FROM documents ORDER BY uploaded_at DESC"))
            return [{"id": row[0], "filename": row[1], "file_hash": row[2]} for row in result]
    except: return []

def remove_document(doc_id, file_hash):
    """Atomically removes doc from registry and vector table"""
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM documents WHERE id = :id"), {"id": doc_id})
            # Correct LlamaIndex table targeting
            actual_vector_table = f"data_{VECTOR_TABLE}"
            conn.execute(text(f"DELETE FROM {actual_vector_table} WHERE metadata_->>'file_hash' = :file_hash"), 
                         {"file_hash": file_hash})
            conn.commit()
    except Exception as e:
        st.error(f"Deletion failed: {e}")

def get_vector_store():
    return PGVectorStore.from_params(
        database=DB_PARAMS["dbname"], host=DB_PARAMS["host"], password=DB_PARAMS["password"],
        port=DB_PARAMS["port"], user=DB_PARAMS["user"], table_name=VECTOR_TABLE, embed_dim=1536
    )

# --- CORE RAG LOGIC ---
Settings.llm = OpenAI(model="gpt-4o", temperature=0.0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

def process_document(file_obj, filename, file_hash):
    # 1. Pre-emptive cleanup of existing hash to avoid 'Zombies'
    engine = get_db_engine()
    with engine.connect() as conn:
        existing = conn.execute(text("SELECT id FROM documents WHERE file_hash = :h"), {"h": file_hash}).fetchone()
        if existing:
            remove_document(existing[0], file_hash)

    # 2. Parse with corrected timeouts
    with st.spinner(f"Parsing {filename}..."):
        temp_path = f"temp_{file_hash}.pdf"
        with open(temp_path, "wb") as f:
            f.write(file_obj.getbuffer())
        
        try:
            parser = LlamaParse(
                result_type="markdown", verbose=True,
                job_timeout_in_seconds=3600, 
                job_timeout_extra_time_per_page_in_seconds=10
            )
            llama_docs = parser.load_data(temp_path)
            
            for d in llama_docs:
                d.metadata.update({"file_hash": file_hash, "filename": filename})

            # 3. Store in Postgres
            vector_store = get_vector_store()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            VectorStoreIndex.from_documents(llama_docs, storage_context=storage_context)
            
            # 4. Register in DB
            with engine.connect() as conn:
                conn.execute(text("INSERT INTO documents (filename, file_hash) VALUES (:n, :h)"),
                             {"n": filename, "h": file_hash})
                conn.commit()
            st.success(f"Processed: {filename}")
        finally:
            if os.path.exists(temp_path): os.remove(temp_path)

# --- UI LAYOUT ---
st.title("CFO Financial Assistant")
init_database()

with st.sidebar:
    st.header("1. Upload Documents")
    files = st.file_uploader("Upload PDF", type=['pdf'], accept_multiple_files=True)
    if files:
        for f in files:
            f_hash = get_file_hash(bytes(f.getbuffer()))
            # Only process if not already there or to overwrite
            process_document(f, f.name, f_hash)

    st.header("2. Document Repository")
    current_docs = get_documents_list()
    if current_docs:
        for i, d in enumerate(current_docs):
            c1, c2 = st.columns([0.8, 0.2])
            c1.text(d['filename'][:20] + "..." if len(d['filename']) > 20 else d['filename'])
            if c2.button("X", key=f"del_{d['id']}_{i}"):
                remove_document(d['id'], d['file_hash'])
                st.rerun()
    else:
        st.info("No documents found.")

# --- CHAT INTERFACE ---
docs_in_db = get_documents_list()
if docs_in_db:
    index = VectorStoreIndex.from_vector_store(get_vector_store())
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_engine" not in st.session_state or st.session_state.get("count") != len(docs_in_db):
