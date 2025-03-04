import streamlit as st
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

# Define State Type for LangGraph
class State(TypedDict):
    messages: Annotated[list, add_messages]

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

def product_owner_review(state: State):
    review_prompt = """
    You are an experienced Agile Product Owner reviewing the generated user stories for a software development project.
    
    Based on your assessment, follow these decision rules:
    - If the user stories need improvement, provide specific feedback and direct the process to **1**.
    - If the user stories are clear, structured, and meet acceptance criteria, approve them and move forward to **0**.
    
    **Project Details:**
    - Project Name: {project_name}
    - Project Description: {project_description}
    - Key Features: {features}
    
    **User Stories for Review:**
    {generated_user_stories}
    
    **Review Criteria:**
    - Clarity & Conciseness
    - Role-Action-Benefit Format
    - Acceptance Criteria Completeness
    - Coverage of Key Features
    
    Decision:
    - If feedback is required, provide clear suggestions and direct to **1**.
    - If approved, confirm Agile compliance and proceed to **0**.
    """
    llm = ChatOpenAI(model="gpt-4o")
    return {"messages":[llm.invoke([review_prompt]+state["messages"])]}


def revise_user_stories(state: State):
    revise_prompt = '''
    
    You are an expert Agile Product Owner refining user stories based on feedback. Your goal is to enhance clarity, completeness, and alignment with Agile principles before approval.

    ### **Project Details:**
    - **Project Name:** {project_name}
    - **Project Description:** {project_description}
    - **Key Features:** {features}

    ### **Initial User Stories:**
    {generated_user_stories}

    ### **Feedback from Product Owner:**
    {product_owner_feedback}

    ### **Revision Guidelines:**
    - Ensure all user stories follow the standard format:  
      **"As a [user role], I want [feature or action] so that [benefit or reason]."**
    - Improve clarity, removing ambiguities and redundant information.
    - Ensure each story has **3-5 acceptance criteria** with clear bullet points.
    - Address any missing details, dependencies, or edge cases.
    - Maintain consistency in tone and format across all stories.

    ### **Deliverables:**
    Revise the provided user stories based on the feedback and present the improved version below.'''

    llm = ChatOpenAI(model="gpt-4o")
    return {"messages":[llm.invoke([revise_prompt]+state["messages"])]}

# Review Decision Function
def review_decision(state: State):
    last_message = state["messages"][-1].content
    if "feedback" in last_message or "revise" in last_message:
        return "1"  # Needs revision
    return "0"  # Approved

# Build the LangGraph Workflow
builder = StateGraph(State)
builder.add_node("generate_user_stories", generate_user_stories)
builder.add_node("product_owner_review", product_owner_review)
builder.add_node("revise_user_stories", revise_user_stories)

builder.add_edge(START, "generate_user_stories")
builder.add_edge("generate_user_stories", "product_owner_review")
builder.add_conditional_edges("product_owner_review", review_decision, {"1": "revise_user_stories", "0": END})
builder.add_edge("revise_user_stories", "generate_user_stories")

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
         # Expander for Intermediate Steps
        with st.expander("📜 Intermediate Steps: User Story Generation Process"):
            for message in result["messages"]:
                st.write(message.content)

        # Expander for Final Refined User Stories
        with st.expander("✅ Final Refined User Stories"):
            final_story = result["messages"][-1].content
            st.write(final_story)

    else:
        st.error("⚠️ Please provide all inputs before generating user stories.")

