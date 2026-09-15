# 📚 Document RAG Assistant

A simple Retrieval-Augmented Generation (RAG) application that allows users to ask questions about documents and receive answers grounded in the retrieved document content.

The project combines **semantic search** with a **Large Language Model (LLM)** to create a document question-answering system.

## 🚀 Features

- 📄 Extract text from DOCX documents
- ✂️ Split documents into smaller chunks
- 🧠 Generate semantic embeddings using Sentence Transformers
- 🔎 Search relevant document chunks using FAISS
- 🤖 Generate grounded answers using a Hugging Face LLM
- 💬 Interactive question-answering through Streamlit
- 📑 Display retrieved source chunks
- ⚡ Cache the RAG system in Streamlit for faster repeated queries
- 🔐 Keep API credentials private using environment variables

## 🏗️ Architecture

```text
                Document
                   │
                   ▼
            Text Extraction
                   │
                   ▼
             Text Chunking
                   │
                   ▼
          Sentence Transformers
             Embeddings
                   │
                   ▼
                FAISS
          Vector Similarity Search
                   │
                   ▼
           Relevant Chunks
                   │
                   ▼
          Hugging Face LLM
                   │
                   ▼
              Final Answer
                   │
                   ▼
             Streamlit UI