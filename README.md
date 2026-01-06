# Agentic RAG using FLAN-T5 
# Agentic RAG Pipeline with Local LLMs

A privacy-focused, local Agentic Retrieval-Augmented Generation (RAG) system built with Python and LangChain. 

## 🤖 What makes this "Agentic"?
Unlike standard RAG systems that search a database for every single query, this system uses an **Agentic Router**. The agent analyzes the user's intent first:
- **Direct Response:** For greetings or general knowledge, the agent answers immediately using its internal weights.
- **Document Search:** For specific queries about your data, the agent triggers a retrieval process from the vector database.



## 🛠️ Tech Stack
- **Framework:** LangChain
- **Orchestration:** Python
- **Vector Database:** ChromaDB
- **Embeddings:** `all-MiniLM-L6-v2` (Sentence-Transformers)
- **LLM:** `google/flan-t5-base` (Optimized for speed and local execution via Hugging Face)
- **Frontend:** Streamlit

##Comaparison Table
-This comparison highlights how an agent-driven approach improves efficiency, privacy, and user intent handling over traditional RAG pipelines.
| Feature                  | Standard RAG                                                                   | Agentic RAG (This Project)                                                                             |
| ------------------------ | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| **Search Logic**         | Always queries the vector database, even for simple inputs like greetings      | **Intelligent Routing** – retrieves documents only when the query actually requires external knowledge |
| **Privacy**              | Often relies on cloud-based APIs (e.g., OpenAI, Claude), risking data exposure | **100% Local Execution** – all data and inference stay on the user’s machine                           |
| **Intent Understanding** | Limited or no explicit intent analysis                                         | **Context-Aware** – analyzes user intent before deciding whether to retrieve or respond directly       |
| **Efficiency & Latency** | Higher latency due to unnecessary retrieval steps                              | **Optimized Performance** – instant responses for non-document queries                                 |



## 🚀 Getting Started

### Prerequisites
- Python 3.10+

### Installation
1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   .\venv\Scripts\activate     # Windows
