import streamlit as st
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

# Define State Type for LangGraph
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Function to Generate User Stories
def generate_user_stories(state:State):
    user_story_prompt = """
    You are an expert Agile product owner specializing in user story generation. Your goal is to create well-structured user stories 
    that align with Agile best practices and ensure clarity for development teams. Each user story must adhere to the following:
    
    - Clearly define the user role.
    - Describe the action or feature the user needs.
    - Specify the benefit or reason behind the feature.
    - Include three to five well-defined acceptance criteria.
    - Ensure that the user stories are clear, concise, and aligned with Agile methodologies.
    
    **Project Details:**
    - Project Name: project_name
    - Project Description: project_description
    - Key Features & Requirements:
    features
    
    **Deliverables:**
    Generate at least one user story per key feature, ensuring:
    - The stories follow the standard format: "As a [user role], I want [feature or action] so that [benefit or reason]."
    - Each story has 3-5 acceptance criteria, formatted with bullet points.
    - If applicable, include edge cases or dependencies.
    
    Please generate the structured user stories below:
    """
    
    llm = ChatOpenAI(model="gpt-4o")
    return {"messages":[llm.invoke([user_story_prompt]+state["messages"])]}

# Build LangGraph Workflow
builder = StateGraph(State)
builder.add_node("generate_user_stories", generate_user_stories)
builder.add_edge(START, "generate_user_stories")
builder.add_edge("generate_user_stories", END)
graph = builder.compile()

# Streamlit UI
st.title("📝 Agile User Story Generator")

# User Input Fields
project_name = st.text_input("Project Name", placeholder="Enter your project name")
project_description = st.text_area("Project Description", placeholder="Provide a brief description of your project")
features = st.text_area("Key Features", placeholder="List features, separated by commas")

# Generate User Stories Button
if st.button("Generate User Stories"):
    if project_name and project_description and features:
        initial_state = {
            "messages": [
                {
                    "role": "system",
                    "content": f"""
                    Project Name: {project_name}
                    Project Description: {project_description}
                    Key Features: {features}
                    """
                }
            ]
        }

        # Invoke LangGraph Workflow
        result = graph.invoke(initial_state)

        # Display Output
        st.subheader("📌 Generated User Stories")
        st.write(result["messages"][1].content)

    else:
        st.error("⚠️ Please provide all inputs before generating user stories.")

