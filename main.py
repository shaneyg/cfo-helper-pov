import streamlit as st
import os
import nest_asyncio
import hashlib
import requests
from datetime import datetime
from urllib.parse import urlparse
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

nest_asyncio.apply()

st.set_page_config(page_title="CFO Helper",
                   page_icon="💰",
                   layout="wide",
                   initial_sidebar_state="expanded")

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
    """,
            unsafe_allow_html=True)

st.title("CFO Financial Assistant")

required_vars = ["OPENAI_API_KEY", "NEON_DATABASE_URL"]
missing_vars = [var for var in required_vars if not os.environ.get(var)]
if missing_vars:
    st.error(f"Missing environment variables: {', '.join(missing_vars)}")
    st.stop()

DATABASE_URL = os.environ.get("NEON_DATABASE_URL",
                              "").strip().replace("\\n", "").strip()
VECTOR_TABLE = "document_vectors"

parsed_url = urlparse(DATABASE_URL)
DB_USER = parsed_url.username
DB_PASSWORD = parsed_url.password
DB_HOST = parsed_url.hostname
DB_PORT = str(parsed_url.port or 5432)
DB_NAME = parsed_url.path.lstrip('/')


@st.cache_resource
def get_db_engine():
    return create_engine(DATABASE_URL,
                         pool_pre_ping=True,
                         pool_recycle=300,
                         connect_args={
                             "connect_timeout": 10,
                             "sslmode": "require"
                         })


def check_database_connection():
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, None
    except OperationalError as e:
        error_msg = str(e)
        if "could not translate host name" in error_msg or "Name or service not known" in error_msg:
            return False, "Database service is temporarily unavailable. This usually resolves itself in a few moments."
        elif "connection refused" in error_msg.lower():
            return False, "Database is not accepting connections. Please wait a moment and try again."
        elif "timeout" in error_msg.lower():
            return False, "Database connection timed out. Please try again."
        else:
            return False, "Unable to connect to the database. Please try again in a moment."
    except Exception as e:
        return False, "Database connection error. Please try again."


def init_database():
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    file_hash VARCHAR(64) NOT NULL UNIQUE,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS visitor_logs (
                    id SERIAL PRIMARY KEY,
                    ip_address VARCHAR(45) NOT NULL,
                    country VARCHAR(100),
                    visited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
        return True
    except Exception:
        return False


def get_visitor_ip():
    try:
        response = requests.get('https://api.ipify.org?format=json', timeout=5)
        return response.json().get('ip', 'Unknown')
    except:
        return 'Unknown'


def get_country_from_ip(ip_address):
    if ip_address == 'Unknown':
        return 'Unknown'
    try:
        response = requests.get(
            f'http://ip-api.com/json/{ip_address}?fields=country', timeout=5)
        return response.json().get('country', 'Unknown')
    except:
        return 'Unknown'


def log_visitor(ip_address, country):
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT id FROM visitor_logs WHERE ip_address = :ip AND visited_at > NOW() - INTERVAL '1 hour'"
                ), {"ip": ip_address})
            if not result.fetchone():
                conn.execute(
                    text(
                        "INSERT INTO visitor_logs (ip_address, country) VALUES (:ip, :country)"
                    ), {
                        "ip": ip_address,
                        "country": country
                    })
                conn.commit()
    except:
        pass


def get_visitor_logs():
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT ip_address, country, visited_at FROM visitor_logs ORDER BY visited_at DESC LIMIT 50"
                ))
            return [{
                "ip": row[0],
                "country": row[1],
                "visited_at": row[2]
            } for row in result]
    except:
        return []


db_ok, db_error = check_database_connection()
if not db_ok:
    st.warning(db_error)
    if st.button("Retry Connection"):
        st.cache_resource.clear()
        st.rerun()
    st.info(
        "Upload functionality will be available once the database connection is restored."
    )
    st.stop()

if not init_database():
    st.warning(
        "Database tables are being set up. Please wait a moment and refresh.")
    if st.button("Retry"):
        st.rerun()
    st.stop()

if "visitor_logged" not in st.session_state:
    visitor_ip = get_visitor_ip()
    visitor_country = get_country_from_ip(visitor_ip)
    log_visitor(visitor_ip, visitor_country)
    st.session_state.visitor_logged = True

from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_parse import LlamaParse

Settings.llm = OpenAI(model="gpt-4o", temperature=0.0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


def get_vector_store():
    return PGVectorStore.from_params(
        database=DB_NAME,
        host=DB_HOST,
        password=DB_PASSWORD,
        port=DB_PORT,
        user=DB_USER,
        table_name=VECTOR_TABLE,
        embed_dim=1536,
    )


def get_documents_list():
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT id, filename, file_hash, uploaded_at FROM documents ORDER BY uploaded_at DESC"
                ))
            return [{
                "id": row[0],
                "filename": row[1],
                "file_hash": row[2],
                "uploaded_at": row[3]
            } for row in result]
    except Exception:
        return []


def add_document_record(filename, file_hash):
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            conn.execute(
                text(
                    "INSERT INTO documents (filename, file_hash) VALUES (:filename, :file_hash) ON CONFLICT (file_hash) DO NOTHING"
                ), {
                    "filename": filename,
                    "file_hash": file_hash
                })
            conn.commit()
    except Exception as e:
        raise Exception(f"Failed to save document record: {str(e)}")


  def remove_document(doc_id, file_hash):
      try:
          engine = get_db_engine()
          with engine.connect() as conn:
              # 1. Remove the record from the 'documents' list
              conn.execute(text("DELETE FROM documents WHERE id = :id"),
                           {"id": doc_id})
              
              # 2. Target the correct vector table
              # Since VECTOR_TABLE = "document_vectors", the actual table is "data_document_vectors"
              actual_vector_table = f"data_{VECTOR_TABLE}"
              
              try:
                  # We use the file_hash to wipe all chunks associated with this file
                  conn.execute(
                      text(f"DELETE FROM {actual_vector_table} WHERE metadata_->>'file_hash' = :file_hash"), 
                      {"file_hash": file_hash}
                  )
              except Exception as e:
                  # This catches cases where the vector table might not exist yet
                  print(f"Vector cleanup skipped: {e}")
                  
              conn.commit()
      except Exception as e:
          st.error(f"Error removing document: {str(e)}")




def get_file_hash(file_content):
    return hashlib.md5(file_content).hexdigest()


def process_document(file_path, filename, file_hash):
    if "LLAMA_CLOUD_API_KEY" in os.environ:
        parser = LlamaParse(
          result_type="markdown", 
          verbose=True,
          # These are the correct parameters for LlamaCloud jobs 
          job_timeout_in_seconds=3600, 
          job_timeout_extra_time_per_page_in_seconds=10
        )
        documents = parser.load_data(file_path)
    else:
        from llama_index.core import SimpleDirectoryReader
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

    for doc in documents:
        doc.metadata["file_hash"] = file_hash
        doc.metadata["filename"] = filename

    try:
        vector_store = get_vector_store()
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store)
        VectorStoreIndex.from_documents(documents,
                                        storage_context=storage_context)
    except Exception as e:
        raise Exception(
            f"Failed to store document embeddings. Please try again.")

    add_document_record(filename, file_hash)


def get_index():
    vector_store = get_vector_store()
    return VectorStoreIndex.from_vector_store(vector_store)


with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader("Upload Reports (PDF)",
                                      type=['pdf'],
                                      accept_multiple_files=True)

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
                        process_document(temp_path, uploaded_file.name,
                                         file_hash)
                        st.success(f"Added: {uploaded_file.name}")
                        existing_hashes.add(file_hash)
                    except Exception as e:
                        st.error(
                            f"Error processing {uploaded_file.name}: {str(e)}")
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

# FIND THIS SECTION AROUND LINE 353:
st.header("2. Document Repository")
docs = get_documents_list()

if docs:
    # Use 'enumerate' to get a unique index 'i' for every row
    for i, doc in enumerate(docs):
        col1, col2 = st.columns([3, 1])
        with col1:
            display_name = doc['filename'][:25] + "..." if len(
                doc['filename']) > 25 else doc['filename']
            st.text(display_name)
        with col2:
            # We add 'i' to the key to guarantee uniqueness (e.g., del_1_0, del_None_1)
            if st.button("X", key=f"del_{doc.get('id', 'new')}_{i}"):
                remove_document(doc['id'], doc['file_hash'])
                st.rerun()
else:
    st.info("No documents uploaded yet.")

docs = get_documents_list()
if docs:
    try:
        index = get_index()

        if "chat_engine" not in st.session_state or st.session_state.get(
                "doc_count") != len(docs):
            st.session_state.chat_engine = index.as_chat_engine(
                chat_mode="context",
                system_prompt=
                """You are a CFO's assistant analyzing financial documents.

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
            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                response = st.session_state.chat_engine.chat(prompt)
                st.write(response.response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.response
                })

    except Exception as e:
        st.error(
            "Error loading documents. Please refresh the page and try again.")
else:
    st.info("Upload PDF documents in the sidebar to get started.")
