import streamlit as st
import os
import nest_asyncio
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_parse import LlamaParse 

# 1. ALLOW ASYNC LOOPS (Fixes Replit errors)
nest_asyncio.apply()

# 2. SETUP KEYS (This is the FIX: Read directly from Replit's Environment)
# We check if the keys are there to prevent crashing if they are missing.
if "OPENAI_API_KEY" not in os.environ:
    st.error("❌ OpenAI API Key is missing! Please add 'OPENAI_API_KEY' to Replit Secrets.")
    st.stop()

# We don't need to manually set them anymore; Replit does it for us.
# The libraries will find them automatically in os.environ.

# 3. SET THE BRAIN (Using GPT-4o for best math ability)
Settings.llm = OpenAI(model="gpt-4o", temperature=0.0)

# 4. DEFINE THE "SMART READER" (LlamaParse)
PERSIST_DIR = "./storage"

@st.cache_resource(show_spinner=False)
def get_ai_brain(file_path, file_hash):
    from llama_index.core import StorageContext, load_index_from_storage
    
    # Check if we have a saved index for this file
    index_path = f"{PERSIST_DIR}/{file_hash}"
    
    if os.path.exists(index_path):
        with st.spinner("Loading saved index..."):
            storage_context = StorageContext.from_defaults(persist_dir=index_path)
            index = load_index_from_storage(storage_context)
            return index
    
    with st.spinner("Reading document tables (LlamaParse)... This only happens once per document."):
        # Check if LlamaCloud key exists
        if "LLAMA_CLOUD_API_KEY" in os.environ:
            parser = LlamaParse(result_type="markdown", verbose=True)
            documents = parser.load_data(file_path)
        else:
            st.warning("⚠️ LlamaCloud Key missing. Tables might be messy.")
            from llama_index.core import SimpleDirectoryReader
            documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

        index = VectorStoreIndex.from_documents(documents)
        
        # Save the index for future use
        index.storage_context.persist(persist_dir=index_path)
        
        return index

def get_file_hash(file_content):
    import hashlib
    return hashlib.md5(file_content).hexdigest()

# 5. BUILD THE WEBSITE
st.set_page_config(page_title="CFO Helper", page_icon="💰", layout="wide", initial_sidebar_state="expanded")

# Custom Colors (Bloomberg Style - Yellow and Black)
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

st.title("💰 Fertiglobe Financial Assistant")

# Sidebar: Upload
with st.sidebar:
    st.header("1. Upload Report")
    uploaded_file = st.file_uploader("Upload Annual Report (PDF)", type=['pdf'])

# Main Application Logic
if uploaded_file:
    # Get file content and hash
    file_content = uploaded_file.getbuffer()
    file_hash = get_file_hash(bytes(file_content))
    
    # Save file temporarily
    with open("temp_report.pdf", "wb") as f:
        f.write(file_content)

    # Load the brain (will use cached version if available)
    index = get_ai_brain("temp_report.pdf", file_hash)

    # Create Chat Engine with "Strict Math" instructions
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode="context",
                system_prompt="""You are a CFO's assistant.

                RULES:
                1. UNITS: ALWAYS check table headers. If a cell says '5.2' and header says '$m', answer '5.2 Million'.
                2. FORMATTING: Do NOT use LaTeX math mode (do not put $ signs around numbers).
                3. STYLE: Use bullet points for multiple data points. Make it easy for an executive to scan.
                4. CURRENCY: Escape dollar signs like this: \$ or just write 'USD'.
                """
            )

    # Chat UI
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask about the Q3 financials..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            st.session_state.messages.append({"role": "assistant", "content": response.response})

else:
    st.info("👈 Please upload a PDF in the sidebar to start.")