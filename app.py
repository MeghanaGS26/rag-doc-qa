import streamlit as st
from sentence_transformers import SentenceTransformer

from src.config import EMBED_MODEL, INDEX_DIR
from src.generator import answer
from src.ingest import chunks_from_bytes
from src.retriever import MODES, Retriever

MAX_FILES = 5
MAX_MB = 15

st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄")
st.title("📄 Document Q&A with citations")


@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource
def load_saved_retriever():
    """Index built locally with `python -m src.ingest` (not used on the web demo)."""
    return Retriever(embedder=get_embedder())


# ---- sidebar: documents and settings ----
uploaded = st.sidebar.file_uploader("Upload PDFs", type="pdf",
                                    accept_multiple_files=True)
st.sidebar.caption("Uploaded PDFs are processed in memory for your session "
                   "and are not saved by this app.")
mode = st.sidebar.selectbox("Retrieval mode", MODES, index=len(MODES) - 1)
k = st.sidebar.slider("Passages to retrieve (k)", 1, 10, 5)

# ---- choose the retriever ----
retriever = None
if uploaded:
    if len(uploaded) > MAX_FILES or any(f.size > MAX_MB * 1024 * 1024 for f in uploaded):
        st.sidebar.error(f"Please upload at most {MAX_FILES} PDFs of {MAX_MB} MB each.")
        st.stop()
    key = tuple((f.name, f.size) for f in uploaded)
    if st.session_state.get("docs_key") != key:
        with st.spinner("Reading and indexing your PDFs..."):
            chunks = []
            for f in uploaded:
                chunks.extend(chunks_from_bytes(f.name, f.getvalue()))
            if not chunks:
                st.error("No readable text found. Scanned PDFs are not supported.")
                st.stop()
            st.session_state.retriever = Retriever.from_chunks(chunks, get_embedder())
            st.session_state.docs_key = key
    retriever = st.session_state.retriever
    st.sidebar.success(f"Indexed {len(retriever.chunks)} passages")
elif (INDEX_DIR / "faiss.index").exists():
    retriever = load_saved_retriever()  # local use with your own saved index

if retriever is None:
    st.info("Upload one or more text-based PDFs in the sidebar to get started, "
            "then ask questions about them.")
    st.stop()

# ---- ask a question ----
question = st.text_input("Ask a question about your documents")
if question:
    with st.spinner("Searching and answering..."):
        chunks = retriever.search(question, k=k, mode=mode)
        try:
            reply = answer(question, chunks)
        except Exception as e:  # rate limits, bad key, network problems
            st.error(f"The language model could not answer right now: {e}")
            st.stop()
    st.subheader("Answer")
    st.write(reply)
    st.subheader("Sources")
    for i, c in enumerate(chunks, start=1):
        with st.expander(f"[{i}] {c['source']} - page {c['page']}"):
            st.write(c["text"])
