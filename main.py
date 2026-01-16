import streamlit as st
import os
import nest_asyncio
import hashlib
from sqlalchemy import create_engine, text

nest_asyncio.apply()

st.set_page_config(page_title="CFO Helper", page_icon="💰", layout="wide", initial_sidebar_state="expanded")

required_vars = ["OPENAI_API_KEY", "DATABASE_URL", "PGDATABASE", "PGHOST", "PGPASSWORD", "PGPORT", "PGUSER"]
missing_vars = [var for var in required_vars if not os.environ.get(var)]
if missing_vars:
    st.error(f"Missing environment variables: {', '.join(missing_vars)}")
    st.stop()

from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_parse import LlamaParse

Settings.llm = OpenAI(model="gpt-4o", temperature=0.0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

DATABASE_URL = os.environ.get("DATABASE_URL")
VECTOR_TABLE = "document_vectors"

@st.cache_resource
def get_db_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True)

def init_database():
    engine = get_db_engine()
    try:
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
    except Exception as e:
        st.error(f"Database initialization error: {str(e)}")

init_database()

def get_vector_store():
    return PGVectorStore.from_params(
        database=os.environ.get("PGDATABASE"),
        host=os.environ.get("PGHOST"),
        password=os.environ.get("PGPASSWORD"),
        port=os.environ.get("PGPORT"),
        user=os.environ.get("PGUSER"),
        table_name=VECTOR_TABLE,
        embed_dim=1536,
    )

def get_documents_list():
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT id, filename, file_hash, uploaded_at FROM documents ORDER BY uploaded_at DESC"))
            return [{"id": row[0], "filename": row[1], "file_hash": row[2], "uploaded_at": row[3]} for row in result]
    except Exception as e:
        st.error(f"Error fetching documents: {str(e)}")
        return []

def add_document_record(filename, file_hash):
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO documents (filename, file_hash) VALUES (:filename, :file_hash) ON CONFLICT (file_hash) DO NOTHING"), {"filename": filename, "file_hash": file_hash})
            conn.commit()
    except Exception as e:
        st.error(f"Error adding document record: {str(e)}")

def remove_document(doc_id, file_hash):
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM documents WHERE id = :id"), {"id": doc_id})
            vector_table = f"data_{VECTOR_TABLE}"
            conn.execute(text(f"DELETE FROM {vector_table} WHERE metadata_->>'file_hash' = :file_hash"), {"file_hash": file_hash})
            conn.commit()
    except Exception as e:
        st.error(f"Error removing document: {str(e)}")

def get_file_hash(file_content):
    return hashlib.md5(file_content).hexdigest()

def process_document(file_path, filename, file_hash):
    if "LLAMA_CLOUD_API_KEY" in os.environ:
        parser = LlamaParse(result_type="markdown", verbose=True)
        documents = parser.load_data(file_path)
    else:
        from llama_index.core import SimpleDirectoryReader
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    
    for doc in documents:
        doc.metadata["file_hash"] = file_hash
        doc.metadata["filename"] = filename
    
    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    
    add_document_record(filename, file_hash)

def get_index():
    vector_store = get_vector_store()
    return VectorStoreIndex.from_vector_store(vector_store)

st.markdown("""
    <style>
    .stApp { background-color: #000000 !important; }
    h1, h2, h3 { color: #FFB900 !important; }
    p, span, label { color: #ffffff !important; }
    .stChatInput { border-color: #FFB900 !important; background-color: #1a1a1a !important; }
    .stChatInput input { color: #ffffff !important; }
    div[data-testid="stChatMessage"] { background-color: #1a1a1a !important; border: 1px solid #FFB900; border-radius: 10px; }
    div[data-testid="stChatMessage"] p { color: #ffffff !important; }
    .stMarkdown, .stText { color: #ffffff !important; }
    [data-testid="stSidebar"] { background-color: #1a1a1a !important; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color: #FFB900 !important; }
    .stButton button { background-color: #FFB900 !important; color: #000000 !important; border: none !important; }
    .stButton button:hover { background-color: #cc9400 !important; }
    .stFileUploader { border-color: #FFB900 !important; }
    .stAlert { background-color: #1a1a1a !important; border: 1px solid #FFB900 !important; }
    .stAlert p { color: #FFB900 !important; }
    div[data-testid="stFileUploader"] label { color: #ffffff !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("Fertiglobe Financial Assistant")

with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader("Upload Reports (PDF)", type=['pdf'], accept_multiple_files=True)
    
    if uploaded_files:
        existing_docs = get_documents_list()
        existing_hashes = {doc["file_hash"] for doc in existing_docs}
        
        for uploaded_file in uploaded_files:
            file_content = uploaded_file.getbuffer()
            file_hash = get_file_hash(bytes(file_content))
            
            if file_hash not in existing_hashes:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    temp_path = f"temp_{file_hash}.pdf"
                    with open(temp_path, "wb") as f:
                        f.write(file_content)
                    try:
                        process_document(temp_path, uploaded_file.name, file_hash)
                        st.success(f"Added: {uploaded_file.name}")
                        existing_hashes.add(file_hash)
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
    
    st.header("2. Document Repository")
    docs = get_documents_list()
    
    if docs:
        for doc in docs:
            col1, col2 = st.columns([3, 1])
            with col1:
                display_name = doc['filename'][:25] + "..." if len(doc['filename']) > 25 else doc['filename']
                st.text(display_name)
            with col2:
                if st.button("X", key=f"del_{doc['id']}"):
                    remove_document(doc['id'], doc['file_hash'])
                    st.rerun()
    else:
        st.info("No documents uploaded yet.")

docs = get_documents_list()
if docs:
    try:
        index = get_index()
        
        if "chat_engine" not in st.session_state or st.session_state.get("doc_count") != len(docs):
            st.session_state.chat_engine = index.as_chat_engine(
                chat_mode="context",
                system_prompt="""You are a CFO's assistant analyzing financial documents.

RULES:
1. UNITS: ALWAYS check table headers. If a cell says '5.2' and header says '$m', answer '5.2 Million'.
2. FORMATTING: Do NOT use LaTeX math mode (do not put $ signs around numbers).
3. STYLE: Use bullet points for multiple data points. Make it easy for an executive to scan.
4. CURRENCY: Escape dollar signs like this: USD or write the currency name.
5. When answering, mention which document the information comes from if relevant.
""")
            st.session_state.doc_count = len(docs)
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        if prompt := st.chat_input("Ask about your financial documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                response = st.session_state.chat_engine.chat(prompt)
                st.write(response.response)
                st.session_state.messages.append({"role": "assistant", "content": response.response})
    
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        st.info("Try uploading a document to get started.")
else:
    st.info("Upload PDF documents in the sidebar to get started.")
