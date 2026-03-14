# chain.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from prompt.agent_prompts import Planner_Prompt, Execution_Prompt, Summarizer_Prompt

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

planner_prompt = ChatPromptTemplate.from_messages([("system", Planner_Prompt), ("human", "{messages}")])
planner_chain = planner_prompt | llm | StrOutputParser()

execution_prompt = ChatPromptTemplate.from_messages([("system", Execution_Prompt), ("human", "{messages}")])
execution_chain = execution_prompt | llm.bind_tools([
    "download_hf_model", "convert_hf_to_gguf", "quantize_gguf", "upload_model_hf"
])

summarizer_prompt = ChatPromptTemplate.from_messages([("system", Summarizer_Prompt), ("human", "{messages}")])
summarizer_chain = summarizer_prompt | llm | StrOutputParser()

print("✅ All chains loaded")