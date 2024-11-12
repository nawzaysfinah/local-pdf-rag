# 📝 PDF Query Chatbot with Ollama and Streamlit

A Retrieval-Augmented Generation (RAG) model that enables you to upload a PDF and interactively ask questions about its content using **Ollama** and a language model (LLM). This project leverages **Streamlit** to provide a user-friendly web interface for uploading PDFs and chatting with a chatbot that understands the PDF's content.

## 📚 Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🌟 Introduction

This project demonstrates how to build a Streamlit web application that integrates a RAG model using **Ollama** and Python. Users can upload a PDF file, and the app will process it to allow interactive querying of the content via a chatbot interface. The application combines text extraction, semantic search, and natural language generation to provide informative answers to user questions.

## ✨ Features

- 📄 **PDF Upload**: Easily upload PDF files through the web interface.
- 📚 **PDF Text Extraction**: Extracts text from uploaded PDFs using `PyPDF2`.
- 🧩 **Text Chunking**: Splits extracted text into manageable chunks for efficient retrieval.
- 🔍 **Semantic Search**: Uses `Faiss` and `SentenceTransformer` for efficient similarity search.
- 🤖 **Chatbot Interface**: Interact with a chatbot to ask questions about the PDF content.
- 🛠 **Local Deployment**: Runs entirely on your local machine, leveraging Ollama for language generation.
- 💾 **Caching Mechanism**: Utilizes Streamlit's caching to optimize performance.
- 💬 **Conversation History**: Displays the conversation history between you and the chatbot.

---

## ✅ Prerequisites

Before you begin, ensure you have met the following requirements:

- **Python 3.x** installed on your system.
- **Ollama** installed and configured with a compatible language model (e.g., LLaMA 2).
  - Follow the [Ollama Installation Guide](https://www.ollama.ai/docs/installation) if you haven't set it up.
  - Ensure the model you wish to use (e.g., `llama2`) is installed in Ollama.
- Required Python libraries:
  - `streamlit`
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

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

Install the required packages:

```bash
pip install streamlit PyPDF2 sentence-transformers faiss-cpu numpy
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

### 1. Run the Streamlit App

From the project directory, run:

```bash
streamlit run app.py
```

### 2. Access the App

Open your web browser and navigate to `http://localhost:8501`.

### 3. Upload a PDF File

- Click on the "Browse files" button or drag and drop a PDF file into the uploader.
- The app will process the PDF and notify you once it's ready.

### 4. Interact with the Chatbot

- Enter your questions in the text input field labeled "Ask a question about the PDF:".
- The chatbot will generate answers based on the content of the uploaded PDF.

---

## 🖼 Screenshots

![Screenshot 2024-10-15 at 9 58 58 AM](https://github.com/user-attachments/assets/911b4d39-6c16-46aa-8b7b-f528c534024f)

---

## 🛠 Troubleshooting

### **Common Issues**

1. **Browser Errors or App Not Loading**

   - **Cause**: Browser extensions or cached data might interfere with the app.
   - **Solution**:
     - Try accessing the app in incognito/private browsing mode.
     - Disable browser extensions, especially those that modify web content.

2. **`huggingface/tokenizers` Warning**

   - **Solution**: This warning is suppressed by setting the environment variable `TOKENIZERS_PARALLELISM` to `"false"` at the beginning of the script.

     ```python
     import os
     os.environ["TOKENIZERS_PARALLELISM"] = "false"
     ```

3. **`Error: pull model manifest: file does not exist`**

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

4. **Ollama Not Found**

   - **Solution**: Ensure that Ollama is installed and added to your system's PATH environment variable.

5. **Streamlit Errors or Warnings**

   - **Solution**:
     - Ensure you are using the latest version of Streamlit.
     - Run `pip install --upgrade streamlit` to update.
     - Check the terminal for error messages and address them accordingly.

6. **PDF Processing Errors**

   - **Cause**: The PDF might be encrypted or have a format that's difficult to process.
   - **Solution**:
     - Ensure the PDF is not password-protected.
     - Try processing another PDF to see if the issue persists.

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

This project is licensed under the MIT License.
