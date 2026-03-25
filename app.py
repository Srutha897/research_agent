import streamlit as st
import os
from dotenv import load_dotenv
from ddgs import DDGS
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.tools import Tool

# Load API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ---------- Page Config ----------
st.set_page_config(page_title="AI Research Agent", page_icon="🔍", layout="centered")

# ---------- Custom CSS ----------
st.markdown("""
<style>
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Title */
    .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Thinking box */
    .thinking-box {
    background: #1e1e2e;
    border-left: 3px solid #7b2ff7;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.4rem 0;
    color: #ffffff !important;
    font-size: 0.9rem;
}
    
    /* Final answer box */
    .answer-box {
        background: linear-gradient(135deg, #1e2a1e, #1a2e1a);
        border: 1px solid #00c853;
        border-radius: 12px;
        padding: 1.5rem;
        color: #e0ffe0;
        font-size: 1rem;
        line-height: 1.7;
        margin-top: 1rem;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
    background-color: #1e1e2e !important;
    color: #ffffff !important;
    border: 1px solid #7b2ff7 !important;
    border-radius: 8px !important;
    caret-color: white !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: #888888 !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #7b2ff7 !important;
        box-shadow: 0 0 0 1px #7b2ff7 !important;
    }
        
    /* Button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #7b2ff7, #00d4ff);
        color: white;
        font-weight: 700;
        font-size: 1rem;
        border: none;
        border-radius: 8px;
        padding: 0.6rem;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.85;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown('<div class="main-title">🔍 AI Research Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by GPT-3.5 + DuckDuckGo · Thinks before it answers</div>', unsafe_allow_html=True)
# Stats row
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div style='text-align:center; background:#1e1e2e; border-radius:10px; padding:1rem; border: 1px solid #7b2ff7'>
        <div style='font-size:1.8rem'>🧠</div>
        <div style='color:#fff; font-weight:700'>ReAct Pattern</div>
        <div style='color:#888; font-size:0.8rem'>Reason → Act → Observe</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div style='text-align:center; background:#1e1e2e; border-radius:10px; padding:1rem; border: 1px solid #00d4ff'>
        <div style='font-size:1.8rem'>🔍</div>
        <div style='color:#fff; font-weight:700'>Live Web Search</div>
        <div style='color:#888; font-size:0.8rem'>Powered by DuckDuckGo</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div style='text-align:center; background:#1e1e2e; border-radius:10px; padding:1rem; border: 1px solid #00c853'>
        <div style='font-size:1.8rem'>⚡</div>
        <div style='color:#fff; font-weight:700'>GPT-3.5 Turbo</div>
        <div style='color:#888; font-size:0.8rem'>OpenAI reasoning engine</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
# ---------- Agent Setup ----------
def ddgs_search(query):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
        if results:
            return "\n".join([r['body'] for r in results])
        return "No results found"

tools = [
    Tool(
        name="web_search",
        func=ddgs_search,
        description="Useful for searching the internet for current information. Input should be a search query."
    )
]

prompt = PromptTemplate.from_template("""
You are a helpful research assistant.
You have access to the following tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
Question: {input}
Thought: {agent_scratchpad}
""")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_react_agent(llm, tools, prompt)

# ---------- UI ----------
question = st.text_input("", placeholder="e.g. What are the latest AI breakthroughs in 2025?")

if st.button("🚀 Research!"):
    if not question:
        st.warning("Please enter a question first!")
    else:
        # Capture agent steps manually
        with st.spinner("Agent is thinking..."):
            # Store intermediate steps
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=False,
                max_iterations=10,
                handle_parsing_errors=True,
                return_intermediate_steps=True  # ← key change!
            )
            result = agent_executor.invoke({"input": question})

        # Show thinking steps
        steps = result.get("intermediate_steps", [])
        if steps:
            st.markdown("### <span style='color:#ffffff'>🧠 Agent Thinking Process</span>", unsafe_allow_html=True)
            for i, (action, observation) in enumerate(steps):
                st.markdown(f"""
                <div class="thinking-box">
                    <b>Step {i+1}</b><br>
                    🤔 <b>Thought:</b> {action.log.strip()}<br><br>
                    🔍 <b>Searched for:</b> {action.tool_input}<br><br>
                    👁️ <b>Found:</b> {str(observation)[:300]}...
                </div>
                """, unsafe_allow_html=True)

        # Show final answer
        st.markdown("### <span style='color:#ffffff'>✅ Final Answer</span>", unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box">{result["output"]}</div>', unsafe_allow_html=True)
