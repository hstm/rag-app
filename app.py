"""
RAG System with Flask Web Interface, Semantic Chunking, and PDF Upload
Install: pip install flask flask-cors anthropic sentence-transformers faiss-cpu numpy pypdf
Run: python app.py
Then open http://localhost:5000 in your browser
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import anthropic
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple
import os
import re
from pypdf import PdfReader
from io import BytesIO

app = Flask(__name__)
CORS(app)


class RAGSystem:
    def __init__(
        self, anthropic_api_key: str, embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.claude_client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = None
        self.chunks = []
        self.dimension = None

    def semantic_chunk_text(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """
        Split text on semantic boundaries (paragraphs, sentences) rather than fixed positions.
        """
        # Split on double newlines (paragraphs)
        paragraphs = re.split(r"\n\s*\n", text)

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph keeps us under max size, add it
            if len(current_chunk) + len(para) + 2 <= max_chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                # Current chunk is full, save it and start new one
                if current_chunk:
                    chunks.append(current_chunk)

                # If single paragraph is too large, split by sentences
                if len(para) > max_chunk_size:
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    current_chunk = ""

                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                            if current_chunk:
                                current_chunk += " " + sentence
                            else:
                                current_chunk = sentence
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sentence
                else:
                    current_chunk = para

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def add_documents(self, documents: List[str], max_chunk_size: int = 1000):
        """
        Add documents to the RAG system using semantic chunking.
        """
        print("Chunking documents semantically...")
        all_chunks = []
        for doc in documents:
            chunks = self.semantic_chunk_text(doc, max_chunk_size)
            all_chunks.extend(chunks)

        self.chunks = all_chunks
        print(f"Created {len(all_chunks)} semantic chunks")

        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        self.dimension = embeddings.shape[1]

        print("Building FAISS index...")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        print(f"Added {self.index.ntotal} vectors to index")

    def retrieve(
        self, query: str, top_k: int = 5, similarity_threshold: float = None
    ) -> List[Tuple[str, float]]:
        if self.index is None:
            raise ValueError("No documents added.")

        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        search_k = min(top_k * 3, len(self.chunks))
        distances, indices = self.index.search(query_embedding, search_k)

        # Auto-set threshold based on gap in distances if not provided
        if similarity_threshold is None and len(distances[0]) > 1:
            dists = distances[0]
            gaps = [dists[i + 1] - dists[i] for i in range(len(dists) - 1)]
            if gaps:
                max_gap_idx = gaps.index(max(gaps))
                if gaps[max_gap_idx] > dists[max_gap_idx] * 0.3:
                    similarity_threshold = dists[max_gap_idx] + gaps[max_gap_idx] * 0.5

        results = []
        seen_starts = set()

        for idx, distance in zip(indices[0], distances[0]):
            if similarity_threshold is not None and distance > similarity_threshold:
                continue

            chunk = self.chunks[idx]
            chunk_start = chunk[:100].strip()

            if chunk_start not in seen_starts:
                results.append((chunk, float(distance)))
                seen_starts.add(chunk_start)

            if len(results) >= top_k:
                break

        return results

    def query(self, question: str, top_k: int = 5) -> dict:
        print(f"Query: {question}")
        results = self.retrieve(question, top_k)

        # Get indices of retrieved chunks for highlighting
        retrieved_indices = []
        for chunk, _ in results:
            try:
                retrieved_indices.append(self.chunks.index(chunk))
            except ValueError:
                pass

        # Cache ALL chunks in system prompt (stays constant across queries)
        # This way, the cache hits on every subsequent query
        all_chunks_text = "\n\n---\n\n".join(
            [f"[Chunk {i}]\n{chunk}" for i, chunk in enumerate(self.chunks)]
        )

        # Tell Claude which chunks are most relevant
        relevant_chunks_str = ", ".join([str(i) for i in retrieved_indices[:top_k]])

        response = self.claude_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2048,
            system=[
                {
                    "type": "text",
                    "text": "You are a helpful assistant that answers questions based on provided context. If the answer cannot be found in the context, say so clearly.",
                },
                {
                    "type": "text",
                    "text": f"Here is the complete knowledge base:\n\n{all_chunks_text}",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            messages=[
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nMost relevant chunks: {relevant_chunks_str}\n\nFocus primarily on the chunks listed above, but you may reference other chunks if needed. Answer:",
                }
            ],
        )

        # Log cache performance
        usage = response.usage
        cache_stats = {
            "input_tokens": usage.input_tokens,
            "cache_creation_input_tokens": getattr(
                usage, "cache_creation_input_tokens", 0
            ),
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
            "output_tokens": usage.output_tokens,
        }
        print(f"Cache stats: {cache_stats}")

        return {
            "answer": response.content[0].text,
            "sources": [{"text": chunk, "distance": dist} for chunk, dist in results],
            "cache_stats": cache_stats,
        }


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    pdf_file = BytesIO(file_bytes)
    reader = PdfReader(pdf_file)

    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n\n"

    return text

# example documents to load on startup
example_docs = [
    """
    Artificial Intelligence (AI) is transforming healthcare in numerous ways. 
    Machine learning algorithms can now detect diseases from medical images with 
    accuracy rivaling human experts. AI-powered diagnostic tools analyze X-rays, 
    MRIs, and CT scans to identify conditions like cancer, pneumonia, and fractures.
    
    Natural language processing helps extract insights from medical records and 
    research papers. Predictive models forecast patient outcomes and identify 
    high-risk individuals who may benefit from early intervention.
    """,
    """
    Climate change is causing significant impacts on global ecosystems. Rising 
    temperatures are leading to more frequent and severe weather events, including 
    hurricanes, droughts, and floods. The Arctic ice is melting at an alarming rate, 
    contributing to sea level rise that threatens coastal communities.
    
    Carbon emissions from fossil fuels are the primary driver of climate change. 
    Renewable energy sources like solar and wind power offer sustainable alternatives 
    that can help reduce greenhouse gas emissions and mitigate climate impacts.
    """,
    """
    Quantum computing represents a paradigm shift in computational power. Unlike 
    classical computers that use bits (0 or 1), quantum computers use qubits that 
    can exist in multiple states simultaneously through superposition.
    
    This enables quantum computers to solve certain problems exponentially faster 
    than classical computers. Applications include cryptography, drug discovery, 
    optimization problems, and simulating quantum systems.
    """,
]

# Custom exception for configuration errors
class ConfigurationError(Exception):
    """Raised when there's a configuration issue."""
    pass

def get_api_key() -> str:
    """Get API key with proper validation."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not api_key or api_key == "your-api-key-here":
        raise ConfigurationError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Please set it with: export ANTHROPIC_API_KEY='your-key-here'"
        )
    
    return api_key

# Initialize RAG system with proper error handling
rag = None # Will be initialized if API key is valid
try:
    API_KEY = get_api_key()
    rag = RAGSystem(anthropic_api_key=API_KEY)
    
    print("Initializing RAG system with example documents...")
    rag.add_documents(example_docs)
    print("RAG system ready!")
except ConfigurationError as e:
    print(f"\n{'='*60}")
    print("CONFIGURATION ERROR")
    print(f"{'='*60}")
    print(f"\n{str(e)}\n")
    print("To fix this:")
    print("1. Get your API key from: https://console.anthropic.com/")
    print("2. Set it as an environment variable:")
    print("   export ANTHROPIC_API_KEY='your-api-key-here'")
    print("3. Restart the server\n")
    print(f"{'='*60}\n")

# Helper function to check if RAG is configured
def check_rag_configured():
    """Check if RAG system is configured and return error if not."""
    if rag is None:
        return jsonify({
            "success": False,
            "error": "Service not configured",
            "message": "ANTHROPIC_API_KEY environment variable is not set",
            "instructions": "Please set ANTHROPIC_API_KEY and restart the server"
        }), 503
    return None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG System with Claude</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .badge {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            margin-top: 10px;
        }
        
        .card {
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }
        
        .input-section {
            margin-bottom: 20px;
        }
        
        .input-section label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        
        .input-section textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1em;
            resize: vertical;
            transition: border-color 0.3s;
        }
        
        .input-section textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .file-upload {
            border: 2px dashed #667eea;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 15px;
        }
        
        .file-upload:hover {
            background: #f8f9ff;
            border-color: #764ba2;
        }
        
        .file-upload input[type="file"] {
            display: none;
        }
        
        .file-upload-label {
            display: block;
            color: #667eea;
            font-weight: 600;
            cursor: pointer;
        }
        
        .file-name {
            margin-top: 10px;
            color: #666;
            font-size: 0.9em;
        }
        
        .button-group {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        
        button {
            flex: 1;
            padding: 12px 24px;
            font-size: 1em;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn-secondary {
            background: #f0f0f0;
            color: #333;
        }
        
        .btn-secondary:hover {
            background: #e0e0e0;
        }
        
        .btn-danger {
            background: #dc3545;
            color: white;
        }
        
        .btn-danger:hover {
            background: #c82333;
            transform: translateY(-2px);
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            color: #667eea;
            font-weight: 600;
        }
        
        .answer-section {
            margin-top: 20px;
            padding: 20px;
            background: #f8f9ff;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        .answer-section h3 {
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .answer-text {
            line-height: 1.6;
            color: #333;
        }
        
        .answer-text h1, .answer-text h2, .answer-text h3 {
            margin-top: 1em;
            margin-bottom: 0.5em;
            color: #667eea;
        }
        
        .answer-text h1 { font-size: 1.5em; }
        .answer-text h2 { font-size: 1.3em; }
        .answer-text h3 { font-size: 1.1em; }
        
        .answer-text p {
            margin-bottom: 1em;
        }
        
        .answer-text ul, .answer-text ol {
            margin-left: 2em;
            margin-bottom: 1em;
            padding-left: 0.5em;
        }
        
        .answer-text li {
            margin-bottom: 0.5em;
            margin-left: 0;
        }
        
        .answer-text strong {
            color: #667eea;
            font-weight: 600;
        }
        
        .answer-text code {
            background: #f0f0f0;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        
        .answer-text pre {
            background: #f0f0f0;
            padding: 12px;
            border-radius: 6px;
            overflow-x: auto;
            margin-bottom: 1em;
        }
        
        .answer-text pre code {
            background: none;
            padding: 0;
        }
        
        .sources {
            margin-top: 20px;
        }
        
        .sources h4 {
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .source-item {
            background: white;
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 8px;
            border: 1px solid #e0e0e0;
            font-size: 0.9em;
            color: #666;
        }
        
        .status {
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-weight: 500;
        }
        
        .status.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
        }
        
        .tab {
            padding: 10px 20px;
            background: none;
            border: none;
            border-bottom: 3px solid transparent;
            cursor: pointer;
            font-weight: 600;
            color: #666;
            transition: all 0.3s;
        }
        
        .tab.active {
            color: #667eea;
            border-bottom-color: #667eea;
        }
        
        .tab:hover {
            color: #667eea;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 RAG System</h1>
            <p>Powered by Claude & Semantic Chunking</p>
        </div>
        
        <div class="card">
            <div class="tabs">
                <button class="tab active" onclick="switchTab('text')">Text Input</button>
                <button class="tab" onclick="switchTab('pdf')">PDF Upload</button>
            </div>
            
            <div id="text-tab" class="tab-content active">
                <div class="input-section">
                    <label>Add Documents (one per line or separated by blank lines):</label>
                    <textarea id="documents" rows="6" placeholder="Paste your documents here..."></textarea>
                    <div class="button-group">
                        <button class="btn-secondary" onclick="addDocuments()">Add Documents</button>
                    </div>
                </div>
            </div>
            
            <div id="pdf-tab" class="tab-content">
                <div class="input-section">
                    <label>Upload PDF Documents:</label>
                    <div class="file-upload" id="fileUploadArea">
                        <label class="file-upload-label" for="pdfInput">
                            📄 Click to select PDF files
                        </label>
                        <input type="file" id="pdfInput" accept=".pdf" multiple onchange="handleFileSelect(event)">
                        <div id="fileName" class="file-name"></div>
                    </div>
                    <div class="button-group">
                        <button class="btn-secondary" onclick="uploadPDFs()">Upload & Process PDFs</button>
                    </div>
                </div>
            </div>
            
            <div id="status"></div>
            
            <div class="input-section">
                <label>Ask a Question:</label>
                <textarea id="question" rows="3" placeholder="What would you like to know?"></textarea>
                <div class="button-group">
                    <button class="btn-primary" onclick="askQuestion()">Ask Question</button>
                </div>
            </div>
            
            <div id="loading" style="display: none;" class="loading">
                Processing your query...
            </div>
            
            <div id="response"></div>
        </div>
    </div>
    
    <script>
    let selectedFiles = [];
    
    function formatAnswer(text) {
        // Convert **bold** to <strong>
        text = text.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');
        
        // Split into lines for processing
        const lines = text.split('\\n');
        const processed = [];
        let inList = false;
        let listItems = [];
    
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            
            if (!line) {
                // Empty line - close any open list and add line break
                if (inList) {
                    processed.push('<ol>' + listItems.join('') + '</ol>');
                    listItems = [];
                    inList = false;
                }
                processed.push('');
                continue;
            }
        
            // Check if it's a numbered list item
            const numberedMatch = line.match(/^(\\d+)\\.\\s+(.+)$/);
            if (numberedMatch) {
                const content = numberedMatch[2];
                // Check if it has bold label followed by colon
                const boldMatch = content.match(/^<strong>(.*?)<\\/strong>:\\s*(.*)$/);
                if (boldMatch) {
                    listItems.push(`<li><strong>${boldMatch[1]}:</strong> ${boldMatch[2]}</li>`);
                } else {
                    listItems.push(`<li>${content}</li>`);
                }
                inList = true;
                continue;
            }
            
            // Check if it's a bullet point
            const bulletMatch = line.match(/^[-•]\\s+(.+)$/);
            if (bulletMatch) {
                listItems.push(`<li>${bulletMatch[1]}</li>`);
                inList = true;
                continue;
            }
            
            // Not a list item - close any open list first
            if (inList) {
                processed.push('<ol>' + listItems.join('') + '</ol>');
                listItems = [];
                inList = false;
            }
            
            // Check if it's a header (short line ending with colon, likely a section title)
            // Must be under 60 chars and either start with capital or contain bold
            if (line.endsWith(':') && line.length < 60 && 
                (line[0] === line[0].toUpperCase() || line.includes('<strong>'))) {
                processed.push(`<h3>${line.replace(/:$/, '')}</h3>`);
            } else {
                // Regular paragraph
                processed.push(`<p>${line}</p>`);
            }
        }

        // Close any remaining list
        if (inList) {
            processed.push('<ol>' + listItems.join('') + '</ol>');
        }

        return processed.join('');
    }
    
    function switchTab(tab) {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        
        if (tab === 'text') {
            document.querySelector('.tab:first-child').classList.add('active');
            document.getElementById('text-tab').classList.add('active');
        } else {
            document.querySelector('.tab:last-child').classList.add('active');
            document.getElementById('pdf-tab').classList.add('active');
        }
    }
    
    function handleFileSelect(event) {
        selectedFiles = Array.from(event.target.files);
        const fileNameDiv = document.getElementById('fileName');
        
        if (selectedFiles.length > 0) {
            const names = selectedFiles.map(f => f.name).join(', ');
            fileNameDiv.textContent = `Selected: ${names}`;
        } else {
            fileNameDiv.textContent = '';
        }
    }
    
    async function uploadPDFs() {
        const statusDiv = document.getElementById('status');
        
        if (selectedFiles.length === 0) {
            statusDiv.innerHTML = '<div class="status error">Please select PDF files first.</div>';
            return;
        }
        
        const formData = new FormData();
        selectedFiles.forEach(file => {
            formData.append('files', file);
        });
        
        statusDiv.innerHTML = '<div class="status">Uploading and processing PDFs...</div>';
        
        try {
            const response = await fetch('/upload_pdfs', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                statusDiv.innerHTML = `<div class="status success">${data.message}</div>`;
                selectedFiles = [];
                document.getElementById('pdfInput').value = '';
                document.getElementById('fileName').textContent = '';
            } else {
                statusDiv.innerHTML = `<div class="status error">Error: ${data.error}</div>`;
            }
        } catch (error) {
            statusDiv.innerHTML = `<div class="status error">Error: ${error.message}</div>`;
        }
    }
    
    async function addDocuments() {
        const docs = document.getElementById('documents').value;
        const statusDiv = document.getElementById('status');
        
        if (!docs.trim()) {
            statusDiv.innerHTML = '<div class="status error">Please enter some documents.</div>';
            return;
        }
        
        const docArray = docs.split(/\\n\\s*\\n/).filter(d => d.trim());
        
        try {
            const response = await fetch('/add_documents', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({documents: docArray})
            });
            
            const data = await response.json();
            
            if (data.success) {
                statusDiv.innerHTML = `<div class="status success">${data.message}</div>`;
                document.getElementById('documents').value = '';
            } else {
                statusDiv.innerHTML = `<div class="status error">Error: ${data.error}</div>`;
            }
        } catch (error) {
            statusDiv.innerHTML = `<div class="status error">Error: ${error.message}</div>`;
        }
    }
    
    async function askQuestion() {
        const question = document.getElementById('question').value;
        const responseDiv = document.getElementById('response');
        const loadingDiv = document.getElementById('loading');
        
        if (!question.trim()) {
            alert('Please enter a question.');
            return;
        }
        
        loadingDiv.style.display = 'block';
        responseDiv.innerHTML = '';
        
        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: question})
            });
            
            const data = await response.json();
            
            loadingDiv.style.display = 'none';
            
            if (data.answer) {
                // const formattedAnswer = formatAnswer(data.answer);
                const formattedAnswer = marked.parse(data.answer);
                
                let html = `
                    <div class="answer-section">
                        <h3>Answer:</h3>
                        <div class="answer-text">${formattedAnswer}</div>
                    </div>
                `;
                
                if (data.sources && data.sources.length > 0) {
                    html += '<div class="sources"><h4>Sources:</h4>';
                    data.sources.forEach((source, idx) => {
                        html += `<div class="source-item"><strong>Source ${idx + 1}:</strong> ${source.text.substring(0, 200)}...</div>`;
                    });
                    html += '</div>';
                }
                
                responseDiv.innerHTML = html;
            } else {
                responseDiv.innerHTML = `<div class="status error">Error: ${data.error}</div>`;
            }
        } catch (error) {
            loadingDiv.style.display = 'none';
            responseDiv.innerHTML = `<div class="status error">Error: ${error.message}</div>`;
        }
    }
    
    document.getElementById('question').addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            askQuestion();
        }
    });
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

# health check endpoint
@app.route("/health")
def health_check():
    """Check if the service is properly configured."""
    if rag is None:
        return jsonify({
            "status": "error",
            "configured": False,
            "message": "ANTHROPIC_API_KEY not configured",
            "instructions": "Set ANTHROPIC_API_KEY environment variable and restart"
        })
    
    return jsonify({
        "status": "healthy",
        "configured": True,
        "documents_count": len(rag.chunks) if hasattr(rag, 'chunks') else 0
    })

# Update existing routes to check configuration
@app.route("/add_documents", methods=["POST"])
def add_documents():
    error = check_rag_configured()
    if error:
        return error
    
    try:
        data = request.json
        documents = data.get("documents", [])

        if not documents:
            return jsonify({"success": False, "error": "No documents provided"})

        rag.add_documents(documents)

        return jsonify({
            "success": True,
            "message": f"Successfully added {len(documents)} document(s) with {len(rag.chunks)} semantic chunks total",
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/upload_pdfs", methods=["POST"])
def upload_pdfs():
    error = check_rag_configured()
    if error:
        return error
    
    try:
        if "files" not in request.files:
            return jsonify({"success": False, "error": "No files uploaded"})

        files = request.files.getlist("files")
        documents = []

        for file in files:
            if file.filename.endswith(".pdf"):
                pdf_bytes = file.read()
                text = extract_text_from_pdf(pdf_bytes)
                documents.append(text)

        if not documents:
            return jsonify({"success": False, "error": "No valid PDF files found"})

        rag.add_documents(documents)

        return jsonify({
            "success": True,
            "message": f"Successfully processed {len(files)} PDF(s) with {len(rag.chunks)} semantic chunks total",
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/query", methods=["POST"])
def query():
    error = check_rag_configured()
    if error:
        return error
    
    try:
        data = request.json
        question = data.get("question", "")

        if not question:
            return jsonify({"error": "No question provided"})

        result = rag.query(question)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    if rag is None:
        print("\n⚠️  WARNING: Server started without valid API key!")
        print("⚠️  All API endpoints will return errors until configured.")
        print("⚠️  Visit http://localhost:5000/health to check status\n")
    
    print("\n" + "=" * 50)
    print("RAG System Web Interface")
    print("=" * 50)
    print("\nStarting server at http://localhost:5000")
    print("Press CTRL+C to stop\n")
    app.run(debug=True, port=5000, use_reloader=True)