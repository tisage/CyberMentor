# streamlit run app.py

import json
import os
from operator import itemgetter

import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (AIMessagePromptTemplate,
                                    ChatPromptTemplate,
                                    HumanMessagePromptTemplate)
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from streamlit.runtime.scriptrunner import get_script_run_ctx

####################################################################################
# API KEYs

load_dotenv() # load environment variables from .env file

# List of keys to handle
api_keys = [
    'OPENAI_API_KEY',
    # 'HUGGINGFACEHUB_API_TOKEN',
    # 'ANTHROPIC_API_KEY',
    # 'ACTIVELOOP_TOKEN',
    # 'TAVILY_API_KEY',
    # 'SERPAPI_API_KEY',
    # 'GOOGLE_API_KEY',
    # 'GOOGLE_CSE_ID',
    # 'OWM_API_KEY'
]

# Dictionary to store the status of each key
loaded_keys = {}

# Check each API key and handle accordingly
for key in api_keys:
    if key in os.environ:
        loaded_keys[key] = True
    else:
        loaded_keys[key] = False

#######################
# Sidebar UI
# API KEY status
st.sidebar.title("API Key Status")

for key, loaded in loaded_keys.items():
    if loaded:
        st.sidebar.success(f"{key} loaded")
    else:
        st.sidebar.warning(f"{key} not found, please input below")
        # Provide input for missing keys
        user_input = st.sidebar.text_input(f"Enter {key}", type="password")
        if user_input:
            os.environ[key] = user_input
            st.sidebar.success(f"{key} set successfully!")

# LLM config
st.sidebar.title("Select LLM Models", help='Select the LLM models for the agents and tools.')
llm_options = ['gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo'] # candidate LLM models

selected_agent_llm = st.sidebar.selectbox("Agent:", llm_options)
selected_rag_llm = st.sidebar.selectbox("RAG:", llm_options)
selected_tools_llm = st.sidebar.selectbox("Tools:", llm_options)

temperature_llm = st.sidebar.slider(
    "Set the Temperature for all LLM models",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.05,
    help="Temperature controls the randomness of responses. Lower values make the output more deterministic."
)

# Solve OpenMP runtime kernel crashed issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

####################################################################################
# utility functions
#
# Function to load prompts from .txt files
@st.cache_data
def load_prompt(file_name):
    prompt_path = os.path.join("prompt", file_name)
    with open(prompt_path, "r") as file:
        prompt = file.read()
    return prompt

# Function to load and split prompts from a combined .txt file
@st.cache_data
def load_combined_prompt(file_name):
    prompt_path = os.path.join("prompt", file_name)
    with open(prompt_path, "r") as file:
        content = file.read()
    
    # Split content using the delimiter
    system_prompt, human_prompt = content.split("---")
    
    return system_prompt.strip(), human_prompt.strip()


# Function to load and parse examples from a .txt file
@st.cache_data
def load_examples(file_name):
    example_path = os.path.join("prompt", file_name)
    with open(example_path, "r") as file:
        examples = file.read().strip().split("Example ")[1:]  # Split and remove empty first part
    
    example_prompts = []
    for example in examples:
        question = example.split("Question: ")[1].split("Answer:")[0].strip()
        answer = "Answer:" + example.split("Answer:")[1].strip()
        example_prompts.append({"question": question, "answer": answer})
    
    return example_prompts

@st.cache_resource
def initialize_llm(model_name, temperature):
    return ChatOpenAI(model=model_name, temperature=temperature)

####################################################################################
# Cache the loading/creation of the vector stores

@st.cache_resource
def load_vectorstore(db_path, loader_type, loader_args):
    if os.path.exists(db_path):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            db_path, embeddings, allow_dangerous_deserialization=True)
        print(f"vector db loaded at {db_path}")
    else:
        loader = loader_type(*loader_args)
        docs_from_file = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=20)
        docs = text_splitter.split_documents(docs_from_file)

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embedding=embeddings)
        vectorstore.save_local(db_path)
        print(vectorstore.index.ntotal)
        print(f"vector db created at {db_path}")
    
    return vectorstore

####################################################################################
# RAG prompt

ku_system_prompt = load_prompt("kb_ku_system_prompt.txt")
cat_system_prompt = load_prompt("kb_cat_system_prompt.txt")
cp_system_prompt = load_prompt("kb_cp_system_prompt.txt")

#######################
# knowledge unit
# Load and cache the cybersecurity knowledge unit vectorstore
db_path_ku = "db/cyber_ku"
loader_args_ku = ["rag/cyber_ku"]
db_ku = load_vectorstore(db_path_ku, PyPDFDirectoryLoader, loader_args_ku)

retriever_ku = db_ku.as_retriever(
    # search_type="similarity",
    search_kwargs={"k": 3},
)


ku_rag_llm = initialize_llm(selected_rag_llm, temperature_llm)

ku_prompt = ChatPromptTemplate.from_messages([("system", ku_system_prompt)])

ku_rag_chain = (
    {
        "context": itemgetter("question") | retriever_ku,
        "question": itemgetter("question"),
    }
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": ku_prompt | ku_rag_llm, "context": itemgetter("context")}
)

knowledge_unit_tool = ku_rag_chain.as_tool(
    name="kb_knowledge_unit",
    description="This intelligent RAG-based tool provides insights to answer questions about knowledge units in cybersecurity, including sources."
)

#######################
# Catalog Advisor Tool

# Load and cache a University catalogs vectorstore
db_path_cat = "db/mercy_catalogs"
loader_args_ku = ["rag/mercy_catalogs"]
db_cat = load_vectorstore(db_path_cat, PyPDFDirectoryLoader, loader_args_ku)

retriever_cat = db_cat.as_retriever(
    # search_type="similarity",
    search_kwargs={"k": 3},
)

cat_rag_llm = initialize_llm(selected_rag_llm, temperature_llm)

cat_prompt = ChatPromptTemplate.from_messages([("system", cat_system_prompt)])

cat_rag_chain = (
    {
        "context": itemgetter("question") | retriever_cat,
        "question": itemgetter("question"),
    }
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": cat_prompt | cat_rag_llm, "context": itemgetter("context")}
)

catalog_advisor_tool = cat_rag_chain.as_tool(
    name="kb_catalog_advisor",
    description="This intelligent RAG-based tool retrieves information on school cybersecurity programs, \
        including degree options, program requirements, and course registration details, along with the relevant sources.")

#######################
# Career Pathways

db_path_cp = "db/cyber_cp"
loader_args_cp = ["rag/cyber_cp"]
db_cp = load_vectorstore(db_path_cp, PyPDFDirectoryLoader, loader_args_cp)

retriever_cp = db_cp.as_retriever(
    # search_type="similarity",
    search_kwargs={"k": 3},
)

cp_rag_llm = initialize_llm(selected_rag_llm, temperature_llm)

cp_prompt = ChatPromptTemplate.from_messages([("system", cp_system_prompt)])

cp_rag_chain = (
    {
        "context": itemgetter("question") | retriever_cp,
        "question": itemgetter("question"),
    }
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": cp_prompt | cp_rag_llm, "context": itemgetter("context")}
)

career_pathways_tool = cp_rag_chain.as_tool(
    name="kb_career_pathways",
    description="This intelligent RAG-based tool retrieves information on career pathways in cybersecurity, \
        including career exploration, skill assessment, career planning, along with the relevant sources.")

####################################################################################
# Tools

# Loading the combined prompts from .txt files
pseudo_system_prompt, pseudo_human_prompt = load_combined_prompt("tool_pseudo_code_prompt.txt")
code_system_prompt, code_human_prompt = load_combined_prompt("tool_code_prompt.txt")
script_system_prompt, script_human_prompt = load_combined_prompt("tool_script_prompt.txt")
ml_system_prompt, ml_human_prompt = load_combined_prompt("tool_ml_prompt.txt")

#######################
# CodeSolver
# Two chains
pseudo_prompt = ChatPromptTemplate.from_messages(
    [("system", pseudo_system_prompt), ("human", pseudo_human_prompt)]
)

code_prompt = ChatPromptTemplate.from_messages(
    [("system", code_system_prompt), ("human", code_human_prompt)]
)

pseudo_llm = initialize_llm(selected_tools_llm, temperature_llm)

question_to_pseudo = pseudo_prompt | pseudo_llm | StrOutputParser()

code_llm = initialize_llm(selected_tools_llm, temperature_llm)

pseudo_to_code = ({"pseudo_code": question_to_pseudo, 
                   "input": itemgetter("input")} | code_prompt | code_llm | StrOutputParser())

code_tool = pseudo_to_code.as_tool(
    name="CodeSolver",
    description="This is a powerful and intelligent tool designed to solve a coding problem using Python"
)

#######################
# CryptoSolver
# few-shot learning + system prompt

# Data model
class CryptoSolution(BaseModel):
    """Formatted response for math problems related to cybersecurity."""
    analysis: str = Field(description="Analysis of the problem including key topic, knowledge, knowledge unit, and importance")
    solution: str = Field(description="Detailed step-by-step solution to the problem")

crypto_system_human_prompt = load_prompt("tool_crypto_solver_prompt.txt")
crypto_system_prompt, crypto_human_prompt = crypto_system_human_prompt.split("---")

# Load crypto examples from a local JSON file
with open('prompt/example_crypto_solver.json', 'r') as file:
    crypto_examples = json.load(file)

# Create the few-shot prompt template using the loaded examples
few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=crypto_examples,
    example_prompt=HumanMessagePromptTemplate.from_template("{question}") + AIMessagePromptTemplate.from_template("{answer}")
)

# Create the final crypto_prompt template
crypto_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', crypto_system_prompt.strip()),
        few_shot_prompt,
        ('human', crypto_human_prompt.strip())
    ]
)

crypto_llm = initialize_llm(selected_tools_llm, temperature_llm)

crypto_chain = crypto_prompt | crypto_llm.with_structured_output(CryptoSolution)

crypto_tool = crypto_chain.as_tool(
    name="CryptoSolver",
    description="This skill-based tool is designed to solve complex math and cryptography problems related to cybersecurity. \
        It provides detailed analyses and step-by-step solutions to enhance understanding of key concepts."
)

#######################
# ScriptCoder
# Network Anomaly Detection Scripts

# Data model
class ScriptSolution(BaseModel):
    """Formatted response for scripting problems related to cybersecurity and network anomaly detection."""
    
    analysis: str = Field(description="Analysis of the problem including category identification, key cybersecurity concepts, and relevant data sources")
    solution: str = Field(description="Detailed, step-by-step script solution addressing the problem")
    validation: str = Field(description="Verification of the script's validity and assessment of potential risks, including any necessary warning messages")

script_prompt = ChatPromptTemplate.from_messages(
    [("system", script_system_prompt), ("human", script_human_prompt)]
)

script_llm = initialize_llm(selected_tools_llm, temperature_llm)

script_chain = script_prompt | script_llm.with_structured_output(ScriptSolution)

script_tool = script_chain.as_tool(
    name="ScriptCoder",
    description="A powerful and intelligent skills-based tool specifically designed to write scripts using Bash, AWK, Sed, and Perl.\
                It assists in automating tasks, processing logs, detecting patterns, and responding to cybersecurity threats. \
                The tool also includes analysis of the problem, identification of relevant data sources, and validation of the generated script."
)

#######################
# ClassifierML
# Machine Learning Script

# Data model
class MLSolution(BaseModel):
    """Formatted response for machine learning problems related to cybersecurity."""
    analysis: str = Field(description="Analysis of the problem including problem formulation, data processing, and model selection")
    solution: str = Field(description="Detailed step-by-step Python code solution covering the complete machine learning workflow")


ml_prompt = ChatPromptTemplate.from_messages(
    [("system", ml_system_prompt), ("human", ml_human_prompt)]
)

ml_llm = initialize_llm(selected_tools_llm, temperature_llm)

ml_chain = ml_prompt | ml_llm.with_structured_output(MLSolution)

ml_tool = ml_chain.as_tool(
    name="ClassifierML",
    description="This is a powerful and intelligent skill-based tool designed to write Python scripts for machine learning tasks in cybersecurity.\
         It provides detailed analyses and step-by-step solutions to enhance understanding of key concepts."
)

####################################################################################
# toolkits
tools = [catalog_advisor_tool, knowledge_unit_tool, career_pathways_tool,
         code_tool, crypto_tool, script_tool, ml_tool]

model = ChatOpenAI(model=selected_agent_llm, temperature=temperature_llm)

memory = SqliteSaver.from_conn_string(":memory:")

# Define the agent prompt to guide response formatting
agent_prompt = '''Your response should thoroughly detail the output of the tool being used.

- If the tool is RAG-based, include all relevant sources of information. Clearly mention the document URLs, file names, titles, section names or references, and page numbers.
- If the tool is skill-based, ensure the response contains a comprehensive output, including both an "analysis" section and a "step-by-step solution" section.

For clarity and understanding, structure the output in an organized manner:
- Convert mathematical expressions or symbols into plain text. For example, use * for multiplication (·), ^ for superscripts, and => for implications (⇒).'''

@st.cache_resource  # allow agent to use cache session in streamlit
def agent_response():
    agent_executor = create_react_agent(model, tools, state_modifier = agent_prompt, checkpointer=memory)
    return agent_executor

####################################################################################
# UI

# get streamlit session id
def _get_session_id():
    session_id = get_script_run_ctx().session_id
    return session_id

st.title("Welcome to CyberMentor!")

# Brief introduction with a call to action
st.markdown("""
    Embark on a journey to master the world of cyber-security with the help of our AI-driven mentoring chatbot. \
        Whether you're navigating career pathways, solving complex cryptographic problems, \
            or exploring machine learning in cyber-security, we've got you covered.

    📚 **Knowledge Mastery:** Clarify concepts and deepen your understanding with AI-assisted learning.
    
    🚀 **Problem Solving:** Get real-time support for writing scripts, solving cryptographic challenges, and developing machine learning models.

    🔍 **Career Pathways:** Receive tailored advice on building and advancing your career in cyber-security.
    
    📝 **Course Registration:** Simplify your academic journey with guided course selection and registration.

    No matter where you are on your cyber-security journey, our chatbot is here to provide the support and resources you need to succeed.
    """)

st.divider() # Add a horizontal line
####################################################################################
# Layout

# session state to store chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        

# Accept user prompt input
if prompt := st.chat_input("Ask me anything!"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # _, session_id = _get_session()
    session_id = _get_session_id()
    print(session_id)
    
    # use session_id as the graph thread_id
    config = {"configurable": {"thread_id": session_id}}

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            app = agent_response()

            events = app.stream(
                {"messages": [HumanMessage(content=prompt)]}, config, stream_mode="values")

            responses = []
            # Display the response
            for event in events:
                event["messages"][-1].pretty_print() # show details in console
                responses.append(event["messages"][-1])
            
            # print(responses)
            last_response = responses[-1]
            st.markdown(last_response.content)

        st.session_state.messages.append({"role": "assistant", "content": last_response.content})
