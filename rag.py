from docx import Document
import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llm import generate_answer


# Load the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(pdf_path):
    """Extract text from all pages of a PDF."""

    document = fitz.open(pdf_path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text

def extract_text_from_docx(docx_path):
    """Extract text from a DOCX document."""

    document = Document(docx_path)

    text = ""

    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"

    return text

def split_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_embeddings(chunks):
    """Convert text chunks into numerical vectors."""

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )
    

    return embeddings


def create_faiss_index(embeddings):
    """Create a FAISS index using cosine similarity."""

    embeddings = embeddings.astype("float32")

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def search_documents(query, index, chunks, top_k=3):
    """Find the most relevant chunks for a question."""

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(
        query_embedding.astype("float32"),
        top_k
    )

    results = []

    for index_position in indices[0]:

        if index_position != -1:
            results.append(
                chunks[index_position]
            )

    return results


def answer_question(question, index, chunks):
    """Retrieve relevant context and generate an answer."""

    # Retrieve relevant chunks
    relevant_chunks = search_documents(
        question,
        index,
        chunks,
        top_k=3
    )

    # Combine retrieved chunks
    context = "\n\n".join(
        relevant_chunks
    )

    # Generate answer using the LLM
    answer = generate_answer(
        question,
        context
    )

    return answer, relevant_chunks


if __name__ == "__main__":

    print("Starting RAG test...")

        # Load the actual document
    document_path = "documents/ml_lab_manual.docx"

    print("Reading document...")

    text = extract_text_from_docx(document_path)

    print(f"Extracted {len(text)} characters.")

    # Create chunks
    chunks = split_text(text)

    print(f"Created {len(chunks)} chunks.")

    # Create embeddings
    embeddings = create_embeddings(chunks)

    print(
        "Embedding shape:",
        embeddings.shape
    )

    # Create FAISS index
    index = create_faiss_index(
        embeddings
    )

    print("FAISS index created.")

    # Ask a question
    question = input("\nAsk a question about the document: ")

    answer, sources = answer_question(
        question,
        index,
        chunks
    )

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer)

    print("\nRetrieved Sources:")

    for i, source in enumerate(sources, 1):

        print(f"\n--- Source {i} ---")
        print(source[:500])

    print("\nRAG pipeline working successfully!")