import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import PyPDF2
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import subprocess

st.title("📄 PDF Query Chatbot with Ollama")

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        # Step 1: Extract text from PDF
        @st.cache_data
        def extract_text_from_pdf(file):
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            return text

        pdf_text = extract_text_from_pdf(uploaded_file)

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

        # Step 3: Generate embeddings and create index
        @st.cache_data
        def create_embeddings_and_index(chunks):
            embedder = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = embedder.encode(chunks, convert_to_numpy=True)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            return embedder, index

        embedder, index = create_embeddings_and_index(chunks)
    st.success("PDF processed successfully!")

    # Step 4: Chatbot interface
    user_input = st.text_input("Ask a question about the PDF:", key="input")

    if user_input:
        # Retrieve relevant chunks
        def retrieve_relevant_chunks(query, index, chunks, embedder, top_k=5):
            query_embedding = embedder.encode([query], convert_to_numpy=True)
            distances, indices = index.search(query_embedding, top_k)
            relevant_chunks = [chunks[idx] for idx in indices[0]]
            return relevant_chunks

        relevant_chunks = retrieve_relevant_chunks(user_input, index, chunks, embedder)
        retrieved_text = "\n\n".join(relevant_chunks)
        prompt = f"Context:\n{retrieved_text}\n\nQuestion: {user_input}\n\nAnswer:"

        # Generate response using Ollama
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
                if process.returncode != 0:
                    st.error(f"Ollama error: {stderr.strip()}")
                    return ""
                return stdout.strip()
            except FileNotFoundError:
                st.error("Error: Ollama executable not found. Please ensure Ollama is installed and in your PATH.")
                return ""


        with st.spinner("Generating answer..."):
            response = ollama_query(prompt)
        
        # Update conversation history
        st.session_state.conversation.append({"question": user_input, "answer": response})

    # Step 5: Display conversation history
    if 'conversation' in st.session_state:
        for chat in st.session_state.conversation[::-1]:  # Display latest messages at the top
            st.markdown(f"**You:** {chat['question']}")
            st.markdown(f"**Chatbot:** {chat['answer']}")
            st.markdown("---")
else:
    st.info("Please upload a PDF file to start.")
