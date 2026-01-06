import streamlit as st
import os
from processing import initialize_vector_db
from agent import get_answer, agent_controller

# 1. Professional UI Styling & Branding
st.set_page_config(page_title="Agentic RAG", page_icon="🤖", layout="wide")

# Custom CSS for a "Dark Mode" Tech Aesthetic
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #374151;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: File Upload & Stats ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80) # Robot Logo
    st.title("📁 Knowledge Base")
    st.markdown("---")
    
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("🚀 Process Documents"):
        if uploaded_files:
            if not os.path.exists("./data"): os.makedirs("./data")
            for uploaded_file in uploaded_files:
                with open(os.path.join("./data", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success("Files saved!")
            st.cache_resource.clear()
            st.rerun()
        else:
            st.error("Upload a PDF first.")

    # Sidebar Statistics: Metrics
    st.markdown("---")
    st.subheader("📊 System Stats")
    pdf_count = len(os.listdir("./data")) if os.path.exists("./data") else 0
    st.metric(label="Documents Loaded", value=pdf_count)
    if pdf_count > 0:
        st.caption("System ready for document queries.")

# --- MAIN INTERFACE ---
st.title("🤖 Agentic RAG Pipeline")
st.markdown("#### *A local, privacy-friendly AI that thinks before it searches.*")

@st.cache_resource
def setup_rag():
    if not os.path.exists("./data") or not os.listdir("./data"):
        return None
    return initialize_vector_db("./data")

retriever = setup_rag()

# Chat History Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHAT LOGIC ---
if retriever is None:
    st.warning("👋 Welcome! Please upload your PDF documents in the sidebar to begin.")
else:
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("Agent Reasoning...", expanded=True) as status:
                st.write("Analyzing query intent...")
                action = agent_controller(prompt)
                
                if action == "search":
                    st.write("🔍 **Intent:** Document Query. Searching local vector database...")
                else:
                    st.write("🤖 **Intent:** General Conversation. Answering from internal weights...")
                
                response, sources = get_answer(prompt, retriever)
                status.update(label="Response Generated!", state="complete", expanded=False)
            
            st.markdown(response)

            if sources:
                with st.expander("📚 Verified Sources"):
                    for i, doc in enumerate(sources):
                        source_file = os.path.basename(doc.metadata.get('source', 'Unknown PDF'))
                        st.markdown(f"**Source {i+1}** (`{source_file}`):")
                        st.caption(doc.page_content)
                        st.divider()
        
        st.session_state.messages.append({"role": "assistant", "content": response})