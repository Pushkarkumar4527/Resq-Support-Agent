# ResQ -- Autonomous AI Support Agent

> **An enterprise-grade AI Support Agent that combines
> Retrieval-Augmented Generation (RAG), Autonomous Tool Calling, and
> Human-in-the-Loop (HITL) approval for secure enterprise automation.**

```{=html}
```
![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red?style=for-the-badge&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-Agent-success?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Google-Gemini_2.5_Flash-orange?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-purple?style=for-the-badge)
![SQLite](https://img.shields.io/badge/SQLite-Database-lightgrey?style=for-the-badge)

```
```

------------------------------------------------------------------------

# 🚀 Overview

ResQ is an enterprise-inspired autonomous AI support assistant that goes
beyond traditional chatbots.

It intelligently decides whether to:

-   Answer normally
-   Search company knowledge using RAG
-   Execute backend operations
-   Ask for human approval before sensitive actions

The project demonstrates how modern LLM agents can safely interact with
enterprise systems without sacrificing security.

------------------------------------------------------------------------

# ✨ Features

-   🧠 Autonomous AI Agent
-   📚 Retrieval-Augmented Generation (RAG)
-   🔎 Semantic Search using ChromaDB
-   🤖 LangChain Tool Calling
-   🛡️ Human-in-the-Loop Guardrail
-   💬 Multi-turn Conversation Memory
-   🗂 SQLite Ticket Management
-   🔒 SQL Injection Protection
-   ⚡ Google Gemini 2.5 Flash
-   🎨 Premium Streamlit UI

------------------------------------------------------------------------

# 🏗️ Architecture

``` mermaid
flowchart TD
A(User)
B(Streamlit UI)
C(LangChain Agent)
D(Gemini 2.5 Flash)
E(RAG Tool)
F(Ticket Tool)
G(ChromaDB)
H(SQLite)
I(HITL Approval)

A-->B
B-->C
C-->D
D-->E
E-->G
D-->F
F-->I
I-->H
```

------------------------------------------------------------------------

# 🛠 Tech Stack

  Layer             Technology
  ----------------- -------------------------
  Frontend          Streamlit
  LLM               Google Gemini 2.5 Flash
  Framework         LangChain
  Vector Database   ChromaDB
  Embeddings        all-MiniLM-L6-v2
  Database          SQLite
  Language          Python 3.11

------------------------------------------------------------------------

# 📂 Project Structure

``` text
ResQ/
│
├── app.py
├── company_faqs.txt
├── tickets.db
├── chroma_db/
├── requirements.txt
├── .env
└── README.md
```

------------------------------------------------------------------------

# ⚙️ Workflow

1.  User enters a natural language request.
2.  LangChain forwards it to Gemini.
3.  Gemini decides whether to:
    -   Reply directly
    -   Search company FAQs
    -   Create a support ticket
4.  FAQ requests use the RAG pipeline.
5.  Ticket requests trigger the HITL approval UI.
6.  After approval, SQLite stores the ticket securely.

------------------------------------------------------------------------

# 🧠 Retrieval-Augmented Generation

ResQ uses:

-   RecursiveCharacterTextSplitter
-   all-MiniLM-L6-v2 embeddings
-   ChromaDB
-   LangChain Retriever Tool

This prevents hallucinations by grounding responses in company
documentation.

------------------------------------------------------------------------

# 🛡 Human-in-the-Loop (HITL)

Instead of allowing direct database writes:

1.  AI requests ticket creation.
2.  Execution pauses.
3.  User reviews the action.
4.  Confirm → Database updated.
5.  Cancel → Action discarded.

This ensures safe enterprise automation.

------------------------------------------------------------------------

# 🔒 Security

-   Parameterized SQL queries
-   Deterministic temperature (0.0)
-   Explicit tool descriptions
-   Read-only RAG
-   Human approval before writes

------------------------------------------------------------------------

# 📦 Installation

``` bash
git clone https://github.com/yourusername/ResQ.git
cd ResQ

python -m venv .venv
```

Windows

``` bash
.venv\Scripts\activate
```

Linux/macOS

``` bash
source .venv/bin/activate
```

Install dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 🔑 Environment Variables

Create a `.env` file.

``` env
GOOGLE_API_KEY=YOUR_API_KEY
```

------------------------------------------------------------------------

# ▶️ Run

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

# 📚 Dependencies

    streamlit
    langchain
    langchain-google-genai
    langchain-community
    langchain-huggingface
    chromadb
    sentence-transformers
    python-dotenv

------------------------------------------------------------------------

# 🚧 Current Limitations

-   Local SQLite storage
-   Local ChromaDB persistence
-   Monolithic Streamlit architecture

------------------------------------------------------------------------

# 🔮 Future Roadmap

-   PostgreSQL (Supabase/Neon)
-   Pinecone Vector Database
-   FastAPI Backend
-   React/Next.js Frontend
-   Docker
-   Kubernetes
-   Authentication
-   Multi-user support
-   CI/CD
-   Cloud deployment

------------------------------------------------------------------------

# 🎓 Learning Outcomes

-   RAG
-   LangChain Agents
-   Tool Calling
-   Gemini Integration
-   AI Safety
-   Vector Databases
-   Prompt Engineering
-   Session State Management

------------------------------------------------------------------------

# 👨‍💻 Author

**Pushkar Kumar**

AI & Software Engineering Enthusiast

------------------------------------------------------------------------

# 📄 License

MIT License

------------------------------------------------------------------------

⭐ **If you found this project helpful, consider giving it a star!**
