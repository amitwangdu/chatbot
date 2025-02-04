from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
from typing import List, Dict
from transformers import pipeline

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
CHUNK_SIZE = 1000  # Characters per chunk

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("db", exist_ok=True)

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="db")

# Create collection for PDF documents
collection = chroma_client.get_or_create_collection(
    name="pdf_documents",
    metadata={"hnsw:space": "cosine"}
)

# Initialize the sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize augmentation model
augmenter = pipeline("text-generation", model="mistralai/Mixtral-8x7B-Instruct-v0.1")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = ''.join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return ""

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split text into chunks"""
    words = text.split()
    chunks, current_chunk = [], []
    current_size = 0
    
    for word in words:
        current_size += len(word) + 1  # +1 for space
        if current_size > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('file')
    uploaded_files = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            try:
                text = extract_text_from_pdf(file_path)
                if not text:
                    raise ValueError("No text could be extracted from the PDF")
                
                chunks = chunk_text(text)
                embeddings = model.encode(chunks).tolist()
                
                ids = [str(uuid.uuid4()) for _ in chunks]
                metadatas = [{"source": filename, "chunk_size": len(chunk)} for chunk in chunks]
                
                collection.add(
                    documents=chunks,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                
                uploaded_files.append({'filename': filename, 'chunks': len(chunks), 'status': 'success'})
                
            except Exception as e:
                uploaded_files.append({'filename': filename, 'error': str(e), 'status': 'error'})
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)
    
    return jsonify({'files': uploaded_files})

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query')
    prompt = data.get('prompt', '')  # Optional prompt for augmentation
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        query_embedding = model.encode(query).tolist()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted_results = []
        for idx, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            formatted_results.append({
                'document': doc,
                'source': metadata['source'],
                'relevance_rank': idx + 1,
                'similarity_score': 1 - distance
            })
        
        # Augmentation (if prompt is provided)
        if prompt:
            augmented_content = augmenter(f"{prompt}: {formatted_results[0]['document']}", max_length=300)[0]['generated_text']
        else:
            augmented_content = None
        
        return jsonify({
            'results': formatted_results,
            'augmentation': augmented_content
        })
        
    except Exception as e:
        print(f"Search error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)