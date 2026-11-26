# RAG-Intelligent-PDF-Assistant-Copier
A Retrieval-Augmented Generation system for intelligent PDF question answering using Python, Hugging Face and FAISS.
# RAG Intelligent PDF Assistant

A full academic-level Retrieval-Augmented Generation (RAG) system running inside Google Colab.

## Architecture
1. **Document Ingestion**: Extracting layout-friendly raw text from uploaded PDFs page-by-page using PyMuPDF.
2. **Chunking**: Fragmenting texts into 600-character segments with 100-character overlap to preserve semantic context.
3. **Indexing**: Creating high-dimensional dense vector embeddings using `all-MiniLM-L6-v2` and indexing them instantly using FAISS (IndexFlatL2).
4. **Answer Generation**: Prompting Google's latest Gemini 3.8 Flash model using strict context-injection instructions.
5. **Evaluation**: Quantitative validation suite computing Precision@K, Recall@K, and MRR (Mean Reciprocal Rank).

## How to run
1. Connect your Gemini API Key as a secure secret under the name `GOOGLE_API_KEY`.
2. Install the necessary dependencies with: `pip install -r requirements.txt`.
3. Run all cells in the Colab notebook to index your documents and launch the Gradio user interface locally.
