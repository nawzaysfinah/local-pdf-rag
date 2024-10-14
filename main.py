import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import PyPDF2
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import subprocess

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text

pdf_path = '/Users/syaz/Desktop/test.pdf'  # Replace with your PDF file path
pdf_text = extract_text_from_pdf(pdf_path)

# Step 2: Chunk the text
def chunk_text(text, max_chunk_size=500):
    sentences = re.split('(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += ' ' + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

chunks = chunk_text(pdf_text)

# Step 3: Generate embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = embedder.encode(chunks, convert_to_numpy=True)

# Step 4: Index with Faiss
embedding_dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dimension)
index.add(chunk_embeddings)

# Save index and chunks
faiss.write_index(index, 'chunks.index')
np.save('chunks.npy', chunks)

# Step 5: Handle user query
def retrieve_relevant_chunks(query, index, chunks, embedder, top_k=5):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    relevant_chunks = [chunks[idx] for idx in indices[0]]
    return relevant_chunks

query = "Your question here"  # Replace with your query
relevant_chunks = retrieve_relevant_chunks(query, index, chunks, embedder)

# Step 6: Generate answer using Ollama
def ollama_query(prompt, model="llama2"):
    try:
        process = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=prompt)
        if stderr:
            print("Ollama error:", stderr)
        return stdout
    except FileNotFoundError:
        print("Error: Ollama executable not found. Please ensure Ollama is installed and in your PATH.")
        return ""

retrieved_text = "\n\n".join(relevant_chunks)
prompt = f"Context:\n{retrieved_text}\n\nQuestion: {query}\n\nAnswer:"
response = ollama_query(prompt)
print("\nGenerated Answer:\n", response)
