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

uploaded_files = st.file_uploader(
    "Upload a document",
    type=["pdf", "docx"]
    accept_multiple_files=True
)


@st.cache_resource
def load_rag_system(document_path):
    """Load and prepare an uploaded document for RAG."""

    from rag import extract_text_from_document

    text = extract_text_from_document(document_path)

    chunks = split_text(text)

    embeddings = create_embeddings(chunks)

    index = create_faiss_index(embeddings)

    return index, chunks


# Process uploaded document
if uploaded_file:

    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing document..."):
        index, chunks = load_rag_system(uploaded_file.name)

    st.success("Document processed successfully! ✅")

else:

    st.info("Please upload a PDF or DOCX document to begin.")



# Question input
question = st.text_input(
    "Ask a question about the document:"
)


if uploaded_file and question:

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