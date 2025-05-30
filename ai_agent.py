# ai_agent.py

import os
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain.schema import AIMessage

# Load API Keys (Assuming they're set in the environment)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def get_response_from_ai_agent(llm_id, query_list, allow_search, system_prompt, provider):
    # Choose LLM based on provider
    if provider == "Groq":
        llm = ChatGroq(model=llm_id).with_config({"system_message": system_prompt})
    elif provider == "OpenAI":
        llm = ChatOpenAI(model=llm_id).with_config({"system_message": system_prompt})
    else:
        return {"error": "Invalid provider. Choose from 'Groq' or 'OpenAI'."}

    # Conditionally include search tool
    tools = [TavilySearchResults(max_results=2)] if allow_search else []

    # Create the agent
    agent = create_react_agent(model=llm, tools=tools)

    # Use the last message as query
    latest_query = query_list[-1] if query_list else "Hello!"
    state = {"messages": [latest_query]}

    # Invoke the agent
    response = agent.invoke(state)
    messages = response.get("messages", [])

    # Extract the AI response
    ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]
    return {"response": ai_messages[-1]} if ai_messages else {"response": "No response received."}
