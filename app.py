import streamlit as st

from src.generator import answer
from src.retriever import MODES, Retriever

st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄")
st.title("📄 Document Q&A with citations")


@st.cache_resource
def get_retriever():
    return Retriever()


try:
    retriever = get_retriever()
except FileNotFoundError:
    st.error("Index not found. Add PDFs to data/docs and run: python -m src.ingest")
    st.stop()

mode = st.sidebar.selectbox("Retrieval mode", MODES, index=len(MODES) - 1)
k = st.sidebar.slider("Passages to retrieve (k)", 1, 10, 5)

question = st.text_input("Ask a question about your documents")
if question:
    with st.spinner("Searching and answering..."):
        chunks = retriever.search(question, k=k, mode=mode)
        reply = answer(question, chunks)
    st.subheader("Answer")
    st.write(reply)
    st.subheader("Sources")
    for i, c in enumerate(chunks, start=1):
        with st.expander(f"[{i}] {c['source']} - page {c['page']}"):
            st.write(c["text"])
