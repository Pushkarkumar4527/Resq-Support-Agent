import os
import streamlit as st
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

# --- Initialize Environment Variables ---
load_dotenv()

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.agents import AgentAction, AgentFinish

# --- Page Config & CSS ---
st.set_page_config(page_title="ResQ: Support Agent", layout="centered")

css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&display=swap');

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #000000 100%);
    color: #e2e8f0;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Playfair Display', serif;
    color: #f8fafc;
}
.stChatMessage {
    background-color: rgba(30, 41, 59, 0.5) !important;
    border-radius: 10px;
    padding: 15px;
}
.stChatInputContainer {
    border-radius: 12px;
    border: 1px solid #1e293b;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# --- Initialize Session State First ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "intermediate_steps" not in st.session_state:
    st.session_state.intermediate_steps = []
if "pending_action" not in st.session_state:
    st.session_state.pending_action = None
if "current_input" not in st.session_state:
    st.session_state.current_input = ""
if "uploaded_files_log" not in st.session_state:
    # Check if default file exists on boot
    default_log = ["company_faqs.txt (Default)"] if os.path.exists("company_faqs.txt") else []
    st.session_state.uploaded_files_log = default_log

# --- RAG Setup (Core Memory) ---
@st.cache_resource
def setup_rag():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # Load default faqs if they exist and haven't been loaded
    if os.path.exists("company_faqs.txt") and len(vectorstore.get()['ids']) == 0:
        loader = TextLoader("company_faqs.txt")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        vectorstore.add_documents(splits)
    
    retriever = vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "company_faqs",
        "Search for information about company FAQs, billing, support policies, etc."
    )
    return retriever_tool, vectorstore

rag_tool, global_vectorstore = setup_rag()

# --- Sidebar: System Health & Admin ---
with st.sidebar:
    st.title("🛡️ ResQ System")
    
    # --- VISUAL SYSTEM HEALTH DASHBOARD ---
    st.markdown("### 📊 System Health")
    if os.path.exists('tickets.db'):
        st.markdown("**SQL Database:** 🟢 Connected")
    else:
        st.markdown("**SQL Database:** 🟡 Pending Ticket")
        
    if os.path.exists('./chroma_db'):
        st.markdown("**AI Memory (RAG):** 🟢 Active")
    else:
        st.markdown("**AI Memory (RAG):** 🔴 Offline")
        
    st.markdown("---")
    
    # --- KNOWLEDGE BASE LOG ---
    st.markdown("### 📂 Active Knowledge Base")
    if not st.session_state.uploaded_files_log:
        st.caption("No documents loaded.")
    else:
        for file_name in st.session_state.uploaded_files_log:
            st.caption(f"📄 {file_name}")
            
    st.markdown("---")
    
    # --- ADMIN UPLOADER ---
    st.header("⚙️ Enterprise Admin")
    uploaded_file = st.file_uploader("Upload Policy Document", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        with st.spinner("Processing document & updating AI memory..."):
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            if uploaded_file.name.endswith('.pdf'):
                new_loader = PyPDFLoader(temp_file_path)
            else:
                new_loader = TextLoader(temp_file_path)
                
            new_docs = new_loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            new_splits = text_splitter.split_documents(new_docs)
            
            global_vectorstore.add_documents(new_splits)
            
            # Update the visual log
            if uploaded_file.name not in st.session_state.uploaded_files_log:
                st.session_state.uploaded_files_log.append(uploaded_file.name)
            
            os.remove(temp_file_path)
            st.success(f"✅ {uploaded_file.name} ingested successfully!")
            
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption("Engineered by Pushkar Kumar")

# --- Main UI Title ---
st.title("ResQ: Support Agent with Actions")

# --- Actionable Tool (Ticketing) ---
@tool
def create_support_ticket(user_id: str, issue_description: str) -> str:
    """Creates a support ticket for a user when they explicitly want to escalate or raise an issue."""
    timestamp = datetime.now().isoformat()
    conn = sqlite3.connect('tickets.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS support_tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            issue_description TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        INSERT INTO support_tickets (user_id, issue_description, timestamp)
        VALUES (?, ?, ?)
    ''', (user_id, issue_description, timestamp))
    conn.commit()
    conn.close()
    return f"Ticket successfully created for user {user_id}."

tools = [rag_tool, create_support_ticket]

# --- Agent Setup ---
@st.cache_resource
def get_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are ResQ, a highly capable customer support agent. "
                   "Use the company_faqs tool to answer general questions regarding policies. "
                   "Use the create_support_ticket tool ONLY when a user explicitly wants to escalate or raise an issue. "
                   "Always ask for their user ID and issue description if you need to create a ticket and don't have it."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    return create_tool_calling_agent(llm, tools, prompt)

agent = get_agent()

# --- Display Chat History ---
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="🛡️"):
            st.write(msg.content)

# --- Execution Logic ---
def get_tool_result(action: AgentAction):
    if action.tool == "company_faqs":
        return rag_tool.invoke(action.tool_input)
    elif action.tool == "create_support_ticket":
        return create_support_ticket.invoke(action.tool_input)
    return f"Unknown tool: {action.tool}"

def execute_agent_step(user_input):
    try:
        response = agent.invoke({
            "input": user_input,
            "chat_history": st.session_state.chat_history,
            "intermediate_steps": st.session_state.intermediate_steps
        })
        
        if isinstance(response, AgentFinish):
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            st.session_state.chat_history.append(AIMessage(content=response.return_values["output"]))
            st.session_state.intermediate_steps = []
            st.session_state.current_input = ""
            st.rerun()
            
        elif isinstance(response, list):
            all_resolved = True
            for action in response:
                if action.tool == "create_support_ticket":
                    st.session_state.pending_action = action
                    all_resolved = False
                    break 
                else:
                    tool_result = get_tool_result(action)
                    st.session_state.intermediate_steps.append((action, tool_result))
            
            if all_resolved:
                execute_agent_step(user_input)
            else:
                st.rerun()
                
    except Exception as e:
        st.error(f"Error during agent execution: {e}")

# --- HITL Guardrail UI ---
if st.session_state.pending_action:
    action = st.session_state.pending_action
    st.warning(f"⚠️ **Guardrail Intercept:** The agent is attempting to perform a state-changing action:\n\n"
               f"**Action:** Create Support Ticket\n"
               f"**User ID:** {action.tool_input.get('user_id', 'Unknown')}\n"
               f"**Issue:** {action.tool_input.get('issue_description', 'Unknown')}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Confirm Ticket Creation", type="primary"):
            with st.spinner("Executing transaction in SQLite..."):
                tool_result = get_tool_result(action)
                st.toast("✅ Ticket saved to SQLite database!", icon="🎉")
                st.session_state.intermediate_steps.append((action, tool_result))
                st.session_state.pending_action = None
                execute_agent_step(st.session_state.current_input)
    with col2:
        if st.button("Cancel Action"):
            st.session_state.intermediate_steps.append((action, "Action was cancelled by the user."))
            st.session_state.pending_action = None
            execute_agent_step(st.session_state.current_input)
            
else:
    # --- Chat Input ---
    if prompt := st.chat_input("How can I help you today?"):
        st.session_state.intermediate_steps = []
        st.session_state.pending_action = None
        
        with st.chat_message("user", avatar="👤"):
            st.write(prompt)
            
        st.session_state.current_input = prompt
        
        with st.spinner("Routing query & analyzing context..."):
            execute_agent_step(prompt)