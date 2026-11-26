# RAG-Intelligent-PDF-Assistant-Copier

A Retrieval-Augmented Generation (RAG) system for intelligent PDF question answering using Python, Hugging Face, FAISS and a RAG API.

# RAG Intelligent PDF Assistant

An academic project that allows users to upload PDF documents, ask questions and receive answers based on the content of their documents.

## Architecture

1. **Document Ingestion:** Extracting text page by page from PDF files using PyMuPDF.
2. **Chunking:** Dividing extracted text into 600-character segments with a 100-character overlap.
3. **Embeddings:** Generating semantic vector representations using `all-MiniLM-L6-v2` from Sentence Transformers.
4. **Vector Indexing:** Storing and searching embeddings using FAISS (`IndexFlatL2`).
5. **Answer Generation:** Using a RAG API to generate answers based on the retrieved document passages.
6. **User Interface:** Providing an interactive interface using Gradio.
7. **Evaluation:** Measuring retrieval performance with Precision@K, Recall@K and MRR.

## Technologies Used

- Python
- Sentence Transformers
- Hugging Face
- FAISS
- PyMuPDF
- Gradio
- RAG API

## How to Run

1. Open the notebook in Google Colab.
2. Install the required dependencies using `requirements.txt`.
3. Configure the API credentials required by your RAG API.
4. Run the notebook cells in order.
5. Upload one or more PDF documents.
6. Ask questions and receive answers based on the uploaded documents.

## Project Objective

The objective is to build an intelligent document assistant that combines semantic search and retrieval-augmented generation to provide relevant, document-based answers.

## Limitations

- The quality of answers depends on the quality of the retrieved passages.
- Scanned PDFs may require OCR.
- Complex tables and diagrams may not be fully extracted.
- The system depends on the availability of the configured RAG API.
