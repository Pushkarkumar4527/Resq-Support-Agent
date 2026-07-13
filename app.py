import streamlit as st
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

# --- Initialize Environment Variables ---
# This must be run before any LangChain or Google API components are instantiated
load_dotenv()

from langchain_community.document_loaders import TextLoader
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

# --- CSS Styling ---
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
</style>
"""
st.markdown(css, unsafe_allow_html=True)

st.title("ResQ: Support Agent with Actions")

# --- RAG Setup ---
@st.cache_resource
def setup_rag():
    loader = TextLoader("company_faqs.txt")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    retriever = vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "company_faqs",
        "Search for information about company FAQs, billing, support policies, etc."
    )
    return retriever_tool

rag_tool = setup_rag()

# --- Actionable Tool ---
@tool
def create_support_ticket(user_id: str, issue_description: str) -> str:
    """Creates a support ticket for a user when they explicitly want to escalate or raise an issue.
    Returns a success message with the ticket details.
    """
    timestamp = datetime.now().isoformat()
    
    # 1. Connect to the SQLite database (this automatically creates tickets.db if it doesn't exist)
    conn = sqlite3.connect('tickets.db')
    cursor = conn.cursor()
    
    # 2. Create the table schema if it doesn't exist yet
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS support_tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            issue_description TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    
    # 3. Insert the new ticket into the database securely using parameterized queries
    cursor.execute('''
        INSERT INTO support_tickets (user_id, issue_description, timestamp)
        VALUES (?, ?, ?)
    ''', (user_id, issue_description, timestamp))
    
    # 4. Commit the transaction and close the connection
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
                   "Use the company_faqs tool to answer general questions regarding billing, support policies, etc. "
                   "Use the create_support_ticket tool ONLY when a user explicitly wants to escalate or raise an issue. "
                   "Always ask for their user ID and issue description if you need to create a ticket and don't have it."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return agent

agent = get_agent()

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "intermediate_steps" not in st.session_state:
    st.session_state.intermediate_steps = []
if "pending_action" not in st.session_state:
    st.session_state.pending_action = None
if "current_input" not in st.session_state:
    st.session_state.current_input = ""

# --- Display Chat History ---
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

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
            # It's a list of AgentActions
            all_resolved = True
            for action in response:
                if action.tool == "create_support_ticket":
                    st.session_state.pending_action = action
                    all_resolved = False
                    break # Pause execution for confirmation
                else:
                    tool_result = get_tool_result(action)
                    st.session_state.intermediate_steps.append((action, tool_result))
            
            if all_resolved:
                execute_agent_step(user_input)
            else:
                st.rerun()
                
    except Exception as e:
        st.error(f"Error during agent execution: {e}")

# --- Handle Pending Action (Human-in-the-Loop) ---
if st.session_state.pending_action:
    action = st.session_state.pending_action
    
    # We display the warning box inside an expander or block so it sits cleanly below the chat
    st.warning(f"⚠️ **Guardrail Intercept:** The agent is attempting to perform a state-changing action:\n\n"
               f"**Action:** Create Support Ticket\n"
               f"**User ID:** {action.tool_input.get('user_id', 'Unknown')}\n"
               f"**Issue:** {action.tool_input.get('issue_description', 'Unknown')}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Confirm Ticket Creation", type="primary"):
            with st.spinner("Executing action..."):
                tool_result = get_tool_result(action)
                st.session_state.intermediate_steps.append((action, tool_result))
                st.session_state.pending_action = None
                # Use the input that originally triggered this action
                execute_agent_step(st.session_state.current_input)
            
    with col2:
        if st.button("Cancel Action"):
            st.session_state.intermediate_steps.append((action, "Action was cancelled by the user."))
            st.session_state.pending_action = None
            execute_agent_step(st.session_state.current_input)
            
else:
    # --- Chat Input ---
    if prompt := st.chat_input("How can I help you today?"):
        # Explicitly clear out any residual state to avoid cross-turn corruption
        st.session_state.intermediate_steps = []
        st.session_state.pending_action = None
        
        # Display the prompt immediately
        with st.chat_message("user"):
            st.write(prompt)
            
        st.session_state.current_input = prompt
        
        with st.spinner("Thinking..."):
            execute_agent_step(prompt)