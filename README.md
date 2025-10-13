# RAG System with Anthropic API (Claude), Flask Web Interface, Semantic Chunking, and PDF Upload

## Features

- ✅ Semantic chunking
- ✅ PDF upload support
- ✅ Prompt caching (90% cost savings!)
- ✅ Smart relevance filtering
- ✅ Simple Web UI
- ✅ FAISS as vector database
- ✅ Type-safe Flask routes
- ✅ Marked.js for Markdown rendering

## Install

``` python -m venv rag-app ```

``` source rag-app/bin/activate ```

```pip install flask flask-cors anthropic sentence-transformers faiss-cpu numpy pypdf```

## Run

1. ``` export ANTHROPIC_API_KEY='your-api-key' ```

2. ``` python app.py ```

3. ``` Open http://localhost:5000 in your browser```

## Optimizations

You can optimize the startup time by skipping the example documents. Just comment out this line:

``` rag.add_documents(example_docs) ```

## Future enhancements

I'm currently working on a production ready version of this app using [FastAPI](https://fastapi.tiangolo.com/) and [Qdrant](https://qdrant.tech/) as the vector database.
