from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from prompt.planner.py import Planner_Prompt,Execution_Prompt,Summarizer_Prompt

load_dotenv()

planner_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        Planner_Prompt
    ),
    MessagesPlaceholder(variable_name="messages"),
])

execution_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        Execution_Prompt
    ),
    MessagesPlaceholder(variable_name="messages"),
])

summarizer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        Summarizer_Prompt
    ),
    MessagesPlaceholder(variable_name="messages"),
])

## Chains

llm = ChatGroq(model_name="llama-3.1-8b-instant") 

planner_chain     = planner_prompt     | llm
execution_chain   = execution_prompt   | llm
summarizer_chain  = summarizer_prompt  | llm