import os
import streamlit as st
import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama


# ------------------------------
# Utility Functions
# ------------------------------
def load_and_split(file_path):
    """Load PDF and split into text chunks."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=160)
    texts = text_splitter.split_documents(docs)
    return texts


def build_vector_store(texts, embeddings, index_path):
    """Build and save FAISS vector store."""
    embedding_size = len(embeddings.embed_query("hello"))
    index = faiss.IndexFlatL2(embedding_size)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    vector_store.add_documents(texts)

    os.makedirs(index_path, exist_ok=True)
    vector_store.save_local(index_path)
    return vector_store


def load_vector_store(index_path, embeddings):
    """Load FAISS index safely."""
    try:
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return None


def rag_answer(vector_store, query, model_name="qwen3:1.7b"):
    """Generate answer using retrieved context + Ollama model."""
    results = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""
You are a helpful assistant. Use the context to answer clearly.

Context:
{context}

Question:
{query}

Answer:
"""
    model = ChatOllama(model=model_name)
    response = model.invoke(prompt)
    return response.content


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="📘 PDF RAG Chat", layout="wide")
st.markdown("<h1 style='text-align:center;'>🤖 Chat with your PDF using RAG + Ollama</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("📂 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file:
        temp_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Unique index path per PDF
        base_name = os.path.splitext(uploaded_file.name)[0]
        index_path = os.path.join("indexes", f"{base_name}_index")
        os.makedirs("indexes", exist_ok=True)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        with st.spinner("⚙️ Loading or building FAISS index..."):
            vector_store = load_vector_store(index_path, embeddings)
            if vector_store:
                st.success(f"📂 Loaded existing index for {base_name}")
            else:
                st.info(f"⚙️ Creating new index for {base_name}")
                texts = load_and_split(temp_path)
                vector_store = build_vector_store(texts, embeddings, index_path)
                st.success("✅ Index created successfully!")

        st.session_state.vector_store = vector_store

with col2:
    st.subheader("💬 Chat with Document")

    if "vector_store" not in st.session_state:
        st.info("👈 Please upload a PDF first.")
    else:
        query = st.text_input("Ask a question about your PDF:")
        if query:
            with st.spinner("🤔 Generating response..."):
                answer = rag_answer(st.session_state.vector_store, query)
            st.markdown("### 🧠 Answer")
            st.write(answer)
