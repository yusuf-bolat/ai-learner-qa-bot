# 🤖 AI Learning Q&A Bot

An intelligent chatbot designed to help beginner and intermediate programmers learn Artificial Intelligence, Python, and Machine Learning. It uses a custom Q&A dataset, a fine-tuned quantized LLM, and Retrieval-Augmented Generation (RAG) to deliver concise and accurate responses. Runs locally using Ollama with optional LangChain and Streamlit integration.

---

## 📚 Features

- 💬 Ask AI/ML questions and get accurate, beginner-friendly answers.
- 📁 Retrieval-Augmented Generation (RAG) for knowledge-grounded responses.
- 🧠 Fine-tuned lightweight LLM (`Phi-3 Mini 4-bit`) using the Unsloth library.
- 🔍 Custom curated dataset with 100+ educational Q&A pairs.
- 🧱 Deployable with Ollama for fast, local inference.
- 🌐 Optional web UI using Streamlit and LangChain for interactive exploration.

---

## 🛠️ Tech Stack

| Category            | Tools Used                                          |
|---------------------|-----------------------------------------------------|
| 💡 Model            | Phi-3 Mini 4-bit (bnb)                             |
| 🧪 Fine-Tuning      | Unsloth, PEFT, LoRA, HuggingFace Datasets          |
| 💾 Quantization     | bitsandbytes, 4-bit GGUF export                    |
| 🧠 Deployment       | Ollama (for local model hosting)                   |
| 🔎 RAG Components   | LangChain, ChromaDB                                |
| 🌐 Frontend (optional) | Streamlit (for chat interface)                  |
| 📁 Dataset          | JSON file with 100+ AI/ML beginner Q&A pairs       |
| 🚀 Training Infra   | Google Colab with CUDA GPU support                 |

---

## 📁 Project Structure

ai-learning-qa-bot/
├── json_extraction_dataset_500.json   # Your training Q&A dataset
├── fine_tune_colab.ipynb              # Fine-tuning script in Colab
├── gguf_model/                        # Exported 4-bit GGUF model for Ollama
├── ModelFile                          # Ollama model configuration
├── app.py                             # (Optional) Streamlit app for chatbot UI
└── README.md                          # This file

---

## 🧪 How It Works

1. Fine-tuning: A 4-bit quantized Phi-3 model is fine-tuned on your curated Q&A dataset using LoRA adapters.
2. Export: The fine-tuned model is exported to GGUF format.
3. Ollama Setup: Model is loaded using an Ollama Modelfile with prompt templates and inference parameters.
4. Querying: You can ask questions via command line (Ollama) or through a web UI (Streamlit).
5. (Optional): Use LangChain + ChromaDB for RAG-based long context retrieval.

---

## 🔧 Installation & Setup

### 1. Fine-Tune Model (Colab)
Open `fine_tune_colab.ipynb` and run all cells to:
- Load model
- Prepare dataset
- Fine-tune using Unsloth
- Export to GGUF for Ollama

### 2. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

### 3. Create `Modelfile`
Use the following:

FROM ./unsloth.Q4_K_M.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|end_of_text|>"
PARAMETER stop "<|user|>"

TEMPLATE """<|user|>
{{ .Prompt }}<|assistant|>
"""

SYSTEM """You are a helpful AI assistant specialized in AI and programming education."""

### 4. Run Model
ollama create aiqa -f Modelfile
ollama run aiqa

---

## 🌐 (Optional) Run Streamlit Chat UI

Install dependencies:
pip install streamlit langchain chromadb

Run app:
streamlit run app.py

---

## 🧠 Example Prompt

User: What is overfitting in machine learning?
Bot: Overfitting is when a model learns the training data too well, including its noise, and performs poorly on unseen data.

---

## 📌 Notes

- Trained on 100 curated AI/ML beginner questions
- Works offline, lightweight for laptops
- Easily expandable to new topics by adding new Q&A data

---

## 📜 License

MIT License – free to use and modify.

---

## 👨‍💻 Author

Created by Yusuf Bolat – ChatGPT-assisted project.
