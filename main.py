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
@st.cache_resource(show_spinner=False)
def get_ai_brain(file_path):
    with st.spinner("Reading document tables (LlamaParse)..."):
        # Check if LlamaCloud key exists
        if "LLAMA_CLOUD_API_KEY" in os.environ:
            parser = LlamaParse(result_type="markdown", verbose=True)
            documents = parser.load_data(file_path)
        else:
            st.warning("⚠️ LlamaCloud Key missing. Tables might be messy.")
            from llama_index.core import SimpleDirectoryReader
            documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

        index = VectorStoreIndex.from_documents(documents)
        return index

# 5. BUILD THE WEBSITE
st.set_page_config(page_title="CFO Helper", page_icon="💰")

# Custom Colors (Fertiglobe Branding)
st.markdown("""
    <style>
    .stApp { background-color: #fcfcfc; }
    h1 { color: #0071bc; } 
    .stChatInput { border-color: #2e8540; }
    div[data-testid="stChatMessage"] { background-color: #ffffff; border: 1px solid #ddd; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("💰 Fertiglobe Financial Assistant")

# Sidebar: Upload
with st.sidebar:
    st.header("1. Upload Report")
    uploaded_file = st.file_uploader("Upload Annual Report (PDF)", type=['pdf'])

# Main Application Logic
if uploaded_file:
    # Save file temporarily
    with open("temp_report.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the brain
    index = get_ai_brain("temp_report.pdf")

    # Create Chat Engine with "Strict Math" instructions
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode="context",
            system_prompt="You are a CFO's assistant. CRITICAL: When reading financial tables, ALWAYS check the header for units (e.g. 'Amounts in USD Millions'). If a cell says '5.2' and the header says 'Millions', the answer is '5.2 Million', not '5.2'."
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