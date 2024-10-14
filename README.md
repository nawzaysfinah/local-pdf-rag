Certainly! Below is a **README.md** file for your project, formatted with proper Markdown for GitHub and including emojis where appropriate.

---

# 📝 PDF Query RAG Model with Ollama

A Retrieval-Augmented Generation (RAG) model that enables you to ask questions about the content of a PDF document using **Ollama** and a language model (LLM). This project extracts text from a PDF, indexes it for efficient retrieval, and utilizes an LLM to generate answers based on your queries.

## 📚 Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🌟 Introduction

This project demonstrates how to build a RAG model using **Ollama** and Python to interactively query the content of a PDF document. It combines text extraction, semantic search, and natural language generation to provide informative answers to user questions.

## ✨ Features

- 📄 **PDF Text Extraction**: Extracts text from PDF files using `PyPDF2`.
- 📚 **Text Chunking**: Splits extracted text into manageable chunks for indexing.
- 🔍 **Semantic Search**: Uses `Faiss` and `SentenceTransformer` for efficient similarity search.
- 🤖 **Language Generation**: Generates answers using an LLM via Ollama.
- 🛠 **Easy to Customize**: Modular code structure allows for easy modifications and extensions.

---

## ✅ Prerequisites

Before you begin, ensure you have met the following requirements:

- **Python 3.x** installed on your system.
- **Ollama** installed and configured with a compatible language model (e.g., LLaMA 2).
  - Follow the [Ollama Installation Guide](https://www.ollama.ai/docs/installation) if you haven't set it up.
- Required Python libraries:
  - `PyPDF2`
  - `sentence-transformers`
  - `faiss-cpu`
  - `numpy`

---

## 🛠 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2. Install Python Dependencies

```bash
pip install PyPDF2 sentence-transformers faiss-cpu numpy
```

### 3. Install and Configure Ollama

Ensure Ollama is installed and the desired LLM model is available.

```bash
# Install Ollama (if not already installed)
# Follow instructions at: https://www.ollama.ai/docs/installation

# Pull the desired model (e.g., LLaMA 2)
ollama pull llama2
```

---

## 🚀 Usage

### 1. Place Your PDF File

Copy the PDF file you wish to query into the project directory and update the `pdf_path` variable in the script accordingly.

```python
pdf_path = 'your_file.pdf'  # Replace with your PDF file name
```

### 2. Run the Script

```bash
python rag_ollama_pdf.py
```

### 3. Enter Your Query

When prompted, type the question you have about the PDF content.

```bash
Enter your question: What does the PDF say about AI in healthcare?
```

---

## 🖥 Example

Below is an example of how the script works.

### **rag_ollama_pdf.py**

```python
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

pdf_path = 'your_file.pdf'  # Replace with your PDF file path
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

# Step 5: Handle user query
def retrieve_relevant_chunks(query, index, chunks, embedder, top_k=5):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    relevant_chunks = [chunks[idx] for idx in indices[0]]
    return relevant_chunks

query = input("Enter your question: ")
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
```

### **Sample Output**

```bash
Enter your question: What are the applications of AI in healthcare?

Generated Answer:
 AI in healthcare is used for diagnosis, treatment planning, drug discovery, personalized medicine, and predictive analytics. It helps in analyzing large datasets to find patterns and improve patient outcomes.
```

---

## 🛠 Troubleshooting

### **Common Issues**

1. **`huggingface/tokenizers` Warning**

   - **Solution**: This warning is suppressed by setting the environment variable `TOKENIZERS_PARALLELISM` to `"false"` at the beginning of the script.

   ```python
   import os
   os.environ["TOKENIZERS_PARALLELISM"] = "false"
   ```

2. **`Error: pull model manifest: file does not exist`**

   - **Cause**: Ollama cannot find the specified model.
   - **Solution**:
     - Verify that the model name is correct and matches the installed model.
     - Install the model using Ollama:

     ```bash
     ollama pull llama2
     ```

     - Update the `ollama_query` function with the correct model name:

     ```python
     def ollama_query(prompt, model="llama2"):
         # ...
     ```

3. **Ollama Not Found**

   - **Solution**: Ensure that Ollama is installed and added to your system's PATH environment variable.

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this project, please:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -am 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to customize this README to better fit your project's specifics. If you have any questions or need further assistance, please let me know! 😊
