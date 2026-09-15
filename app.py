import streamlit as st

from rag import (
    extract_text_from_docx,
    split_text,
    create_embeddings,
    create_faiss_index,
    answer_question
)


st.set_page_config(
    page_title="Document RAG Assistant",
    page_icon="📚",
    layout="wide"
)


st.title("📚 Document RAG Assistant")
st.write("Ask questions about your Machine Learning Lab Manual.")


@st.cache_resource
def load_rag_system():
    """Load and prepare the document for RAG."""

    document_path = "documents/ml_lab_manual.docx"

    text = extract_text_from_docx(document_path)

    chunks = split_text(text)

    embeddings = create_embeddings(chunks)

    index = create_faiss_index(embeddings)

    return index, chunks


# Load RAG system
with st.spinner("Loading document..."):
    index, chunks = load_rag_system()


st.success("Document loaded successfully! ✅")


# Question input
question = st.text_input(
    "Ask a question about the document:"
)


if question:

    with st.spinner("Finding the answer..."):

        answer, sources = answer_question(
            question,
            index,
            chunks
        )


    st.subheader("Answer")

    st.write(answer)


    st.subheader("Retrieved Sources")

    for i, source in enumerate(sources, 1):

        with st.expander(f"Source {i}"):

            st.write(source[:500])