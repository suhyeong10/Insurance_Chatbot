# 🛡️ Insurance Chatbot

An intelligent chatbot that provides insurance recommendations and information.  
It analyzes user queries, searches relevant insurance documents, and offers tailored responses.

## 📁 Project Structure

```
Insurance_Chatbot/
├── app.py
├── README.md
└── database/
    ├── documents/
    └── vector_store/
```

## 🧠 Features

- Converts user questions into keyword-based search queries
- Uses RAG (Retrieval-Augmented Generation) for document-grounded responses
- Generates answers via Ollama-powered LLM (`insurance-gemma-3-12b`)
- Maintains session-based chat history
- Enhances response quality with LongContextReorder

## 🚀 How to Run

1. **Install required libraries**

```bash
pip install -r requirements.txt
```

2. Add insurance documents
Place PDF files into the database/documents/ directory.

3. Run Ollama server and download model
Ensure the Ollama server is running locally at `http://localhost:11434`, and install the required model:

```bash
ollama pull codingchild/insurance-gemma-3-12b
```

4. Start the chatbot
```bash
python app.py
```

5. Type your question to receive insurance guidance. Type `exit` to quit.

### 💬 Example Questions
- I'm 30 and I have lung cancer. Can I still get life insurance?
- Can I convert my group insurance after leaving my job?
- What are the differences between term and whole life insurance?
- How does the accelerated death benefit rider work?

### ⚙️ Technologies Used
- Python 3.10
- LangChain
- HuggingFace Transformers
- FAISS
- Ollama
- PyTorch

### 📝 License
This project is freely available for research and non-commercial educational purposes.
For commercial use, please contact the author.
