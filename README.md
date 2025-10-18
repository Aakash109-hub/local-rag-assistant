## 🧠 Local RAG PDF Chatbot using Streamlit + Ollama + LangChain

This project is a **fully local Retrieval-Augmented Generation (RAG)** chatbot that allows you to **upload a PDF** and **ask questions** about its content — all running **offline** on your system.

It uses **LangChain** for retrieval, **FAISS** for vector search, and **Ollama** to run **local LLMs** like `qwen3`, `mistral`, or `llama3` — ensuring **data privacy** and **zero cloud dependency**


<img width="1920" height="1020" alt="Screenshot 2025-10-18 142649" src="https://github.com/user-attachments/assets/4fe985f4-96f9-4a7e-89c3-c1106c455242" />


---

### ⚡ Key Features

* 📄 **Upload any PDF** and interact with its content
* 💾 **Local embeddings & FAISS storage** — no data leaves your machine
* 🤖 **Ollama local LLM integration** — no API keys or internet required
* 🧩 **LangChain-powered retrieval** for accurate context-based answers
* 💬 **Streamlit UI** with two sections:

  * Left → Upload PDF
  * Right → Chat interface
* 🔐 100% **offline, private, and free to run**

---

### 🏗️ Tech Stack


| Component       | Tool                             |
| --------------- | -------------------------------- |
| Language Model  | Ollama (local)                   |
| Framework       | LangChain                        |
| Vector Store    | FAISS                            |
| Embeddings      | HuggingFace (`all-MiniLM-L6-v2`) |
| UI              | Streamlit                        |
| Document Loader | PyPDFLoader                      |

---

### 📂 Project Structure

```
📁 local-rag-chatbot
│
├── app.py                  # Streamlit UI & RAG logic
├── requirements.txt        # Dependencies
├── data/                   # Uploaded PDFs
├── rag_index/              # Local FAISS index (auto-managed)
└── README.md               # Project documentation
```

---

### ⚙️ Installation & Setup

#### 1️⃣ Clone this repository

```bash
git clone https://github.com/Aakash109-hub/local-rag-assistant.git
cd local-rag-assistant
```

#### 2️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate     # (Mac/Linux)
venv\Scripts\activate        # (Windows)
```

#### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

#### 4️⃣ Install Ollama (for local models)

Visit 👉 [https://ollama.ai](https://ollama.ai)
Download and install Ollama for your OS.

#### 5️⃣ Pull your preferred local model

```bash
ollama pull qwen3:1.7b
```

Other supported models:

* `ollama pull mistral`
* `ollama pull llama3`
* `ollama pull phi3`

#### 6️⃣ Run the Streamlit app

```bash
streamlit run app.py
```

---

### 🧠 How It Works

1. **Upload PDF** → The app extracts text using `PyPDFLoader`.
2. **Text Splitting** → Uses `RecursiveCharacterTextSplitter` to create manageable chunks.
3. **Embedding** → Generates embeddings locally using `HuggingFaceEmbeddings`.
4. **Vector Store** → Saves to a **local FAISS index**.
5. **Query** → Retrieves top results using similarity search.
6. **Response** → Sends context and query to the **Ollama local LLM** to generate a natural answer.

🧹 When a new PDF is uploaded, the previous FAISS index is **cleared automatically** to avoid mixing data between files.

---

### 🔐 Why Local?

* 🧾 **No internet required**
* 🧠 **All data and embeddings stay on your machine**
* 💸 **No API costs**
* ⚙️ **Full control** over model, indexing, and storage

This makes it perfect for **private documents, research papers, or company files** you don’t want to send to the cloud.

---

### 🧰 Example Usage

1. Run the app:

   ```bash
   streamlit run app.py
   ```
2. Upload your PDF file (e.g., `research_paper.pdf`)
3. Ask:

   > “What is the main conclusion of this paper?”
4. Get a **context-aware response** generated locally using Ollama.

---

### 📸 UI Overview

| Section         | Description                        |
| --------------- | ---------------------------------- |
| **Left Panel**  | PDF Upload and Index Management    |
| **Right Panel** | Chat Interface and Model Responses |
