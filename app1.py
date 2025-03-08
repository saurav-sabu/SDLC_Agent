import streamlit as st
import pandas as pd
import os
from typing import TypedDict, Annotated, Literal, Optional, List
from pydantic import Field, BaseModel
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langchain_core.messages import *
import tempfile
import base64
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Documentation Generator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the State class
class State(TypedDict):
    messages: Annotated[Optional[List], add_messages]
    project_name: str
    project_description: str
    features: List[str]
    product_decision: str
    feedback: str
    technical_documentation: str
    functional_documentation: str
    combined_documentation: str


# Define the structured output for product owner routing
class ProductOwnerRoute(BaseModel):
    step: Literal["Approved", "Feedback"] = Field(description="The next step in routing process")
    feedback: str = Field(description="If the user stories are not good, provide feedback on how to improve them.")

# Initialize LLM instance
@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o")

llm = get_llm()
router_product_owner_route = llm.with_structured_output(ProductOwnerRoute)

# Functions from your original code
def generate_user_stories(state: State):
    user_story_prompt = f"""
    You are an expert Agile product owner specializing in user story generation. Each user story must:
    - Clearly define the user role.
    - Describe the action or feature needed.
    - Specify the benefit or reason behind the feature.
    - Include 3-5 well-defined acceptance criteria.

    **Project Details:**
    - Name: {state["project_name"]}
    - Description: {state["project_description"]}
    - Features:
    {state["features"]}

    Generate at least one user story per feature, following the format:
    "As a [user role], I want [feature or action] so that [benefit or reason]."

    Ensure clarity, completeness, and Agile best practices.
    """

    messages = state.get("messages", [])
    response = llm.invoke([user_story_prompt] + messages)
    messages.append(response)

    return {"messages": messages}

def product_owner_review(state: State):
    """Routes the user stories for approval or revision."""
    message_content = state["messages"][-1].content  # Extract content from last message
    decision = router_product_owner_route.invoke(
        [
            SystemMessage(content="""Route the input to Approved or Feedback based on user stories quality.
            If 'Approved', leave feedback empty or provide positive reinforcement.
            If 'Feedback', provide constructive feedback on how to improve the stories."""),
            HumanMessage(content=message_content),
        ]
    )
    return {"product_decision": decision.step, "feedback": decision.feedback}

def route_product_decision(state: State):
    """Routes the workflow based on product owner decision."""
    if state["product_decision"] == "Feedback":
        return "Revise User Stories"
    else:  # "Approved"
        return "Generate Technical Documentation"

def revise_user_stories(state: State):
    revise_prompt = f"""
    You are refining user stories based on feedback to ensure clarity, completeness, and Agile alignment.

    **Project Details:**
    - Name: {state["project_name"]}
    - Description: {state["project_description"]}
    - Features:
    {state["features"]}

    **Feedback:**
    {state["feedback"]}

    Original User Stories:
    {state["messages"][-1].content}

    Improve the user stories by:
    - Enhancing clarity and removing ambiguities.
    - Ensuring each story follows the format:
      "As a [user role], I want [feature or action] so that [benefit or reason]."
    - Adding 3-5 clear acceptance criteria.
    - Addressing missing details, dependencies, and edge cases.

    Provide the improved user stories below.
    """

    messages = state.get("messages", [])
    revised_response = llm.invoke([revise_prompt] + messages)
    messages.append(revised_response)

    return {"messages": messages}

def generate_technical_documentation(state: State):
    technical_documentation_prompt = f'''
    You are an expert technical writer skilled in software documentation. Generate a well-structured technical documentation for a project based on the given user stories.

    ### **Project Details**
    - **Project Name**: {state["project_name"]}
    - **Project Description**: {state["project_description"]}
    - **Feature Names**: {state["features"]}

    ### **User Stories**
    {state["messages"][-1].content}

    ### **Expected Documentation Format**
    1. **Introduction**  
       - Briefly describe the purpose of the project.
       - Outline key objectives and goals.

    2. **System Architecture**  
       - List the technologies used and how they interact.
       - Mention integration points (APIs, databases, etc.).

    3. **Features & Functionalities**  
       - For each user story, include:
         - **Feature Name**
         - **User Story**
         - **Acceptance Criteria**
         - **System Behavior**
         - **Edge Cases & Error Handling**

    4. **API Documentation**  
       - List API endpoints, request/response formats, authentication, and error handling.

    5. **Database Schema**  
       - Provide an ERD or describe the database structure.

    6. **Security Considerations**  
       - Define authentication, authorization, encryption, and security policies.

    7. **Performance & Scalability**  
       - Mention caching, load balancing, and system optimizations.

    8. **Deployment & DevOps Strategy**  
       - Outline CI/CD pipelines, infrastructure, and environments.

    9. **Testing Strategy**  
       - Detail unit tests, integration tests, and performance testing.

    10. **Maintenance & Future Enhancements**  
       - Include logging, monitoring, and planned improvements.

    ### **Output Format**
    Generate the documentation in markdown format with proper formatting and bullet points.
    '''
    
    messages = state.get("messages", [])
    response = llm.invoke([technical_documentation_prompt] + messages)
    messages.append(response)
    
    technical_documentation = response.content  # Store the response separately
    
    return {"messages": messages, "technical_documentation": technical_documentation}

def generate_functional_documentation(state: State):
    functional_documentation_prompt = f'''
    You are an expert technical writer skilled in software documentation. Generate a well-structured **Functional Specification Document (FSD)** for a project based on the given user stories.

    ### **Project Overview**
    - **Project Name**: {state["project_name"]}
    - **Project Description**: {state["project_description"]}
    - **Features**: {state["features"]}

    ### **Functional Requirements**
    For each user story, define the functional requirements in detail.

    {state["messages"][-1].content}

    ### **Functional Specification Format**
    1. **Introduction**  
       - Overview of the project and its purpose.  
       - Business needs and problem statement.  
       - Key stakeholders and end users.  

    2. **Scope of the System**  
       - Define what is in scope and out of scope.  
       - List assumptions and dependencies.  

    3. **System Features & Functionalities**  
       For each feature derived from user stories, include:  
       - **Feature Name**:  
       - **User Story**:   
       - **Functional Requirement**:   
       - **Preconditions**: 
       - **Main Flow**: 
       - **Alternate Flows & Edge Cases**: 
       - **Postconditions**: 

    4. **User Interface (UI) Specifications**  
       - Describe UI components and user interactions.  
       - Provide wireframes or mockups (if applicable).  

    5. **Data Flow & Processing**  
       - Explain how data flows through the system.  
       - Define key data transformations and validation rules.  

    6. **Integration Points**  
       - List external systems, APIs, or databases the system interacts with.  
       - Mention data exchange formats and integration protocols.  

    7. **Security & Compliance**  
       - Define authentication, authorization, and access control mechanisms.  
       - Mention data privacy and regulatory compliance considerations.  

    8. **Performance & Scalability Considerations**  
       - Specify system response times, concurrency limits, and scalability strategies.  

    9. **Error Handling & Logging**  
       - Describe error messages, logging levels, and failure recovery strategies.  

    10. **Constraints & Limitations**  
       - Define hardware/software limitations, licensing constraints, etc.  

    11. **Acceptance Criteria & Validation**  
       - Define the criteria to determine feature completion.  
       - Mention functional test cases and expected outputs.  

    ### **Output Format**
    Generate the documentation in **Markdown format** with structured headings, bullet points, and code snippets where necessary.
    '''
    
    messages = state.get("messages", [])
    response = llm.invoke([functional_documentation_prompt] + messages)
    messages.append(response)
    
    functional_documentation = response.content  # Store the response separately
    
    return {"messages": messages, "functional_documentation": functional_documentation}

def get_download_link(content, filename):
    """Generate a download link for a text file."""
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Streamlit app UI
st.title("Documentation Generator")
st.subheader("Generate User Stories and Documentation from Project Details")

# Sidebar for project input
with st.sidebar:
    st.header("Project Details")
    
    project_name = st.text_input("Project Name", key="project_name")
    project_description = st.text_area("Project Description", height=150, key="project_description")
    
    features_input = st.text_area("Features (One per line)", height=150, 
                                  help="Enter one feature per line", key="features_input")
    
    if features_input:
        features_list = [f.strip() for f in features_input.split("\n") if f.strip()]
    else:
        features_list = []
    
    # Display features count
    st.caption(f"Number of features: {len(features_list)}")
    
    generate_button = st.button("Generate Documentation", type="primary", 
                               disabled=not (project_name and project_description and features_list))

# Main content area
if 'results' not in st.session_state:
    st.session_state.results = None

if generate_button:
    with st.spinner("Generating documentation... This may take a few minutes."):
        # Build the graph
        builder = StateGraph(State)

        # Add nodes
        builder.add_node("Generate User Stories", generate_user_stories)
        builder.add_node("Product Owner Review", product_owner_review)
        builder.add_node("Revise User Stories", revise_user_stories)
        builder.add_node("Generate Technical Documentation", generate_technical_documentation)
        builder.add_node("Generate Functional Documentation", generate_functional_documentation)

        # Add edges
        builder.add_edge(START, "Generate User Stories")
        builder.add_edge("Generate User Stories", "Product Owner Review")
        builder.add_edge("Revise User Stories", "Product Owner Review")
        builder.add_edge("Product Owner Review", "Generate Technical Documentation")
        builder.add_edge("Generate Technical Documentation", "Generate Functional Documentation")
        builder.add_edge("Generate Functional Documentation", END)

        # Add conditional edges
        builder.add_conditional_edges(
            "Product Owner Review",
            route_product_decision,
            {
                "Revise User Stories": "Revise User Stories",
                "Generate Technical Documentation": "Generate Technical Documentation"
            }
        )

        # Compile graph
        graph = builder.compile()

        # Initialize state
        initial_state = {
            "project_name": project_name,
            "project_description": project_description,
            "features": features_list,
            "messages": []
        }

        # Execute the graph and get results
        results = graph.invoke(initial_state)
        
        # Store results in session state
        st.session_state.results = results
        
        st.success("Documentation generated successfully!")

# Display results if available
if st.session_state.results:
    results = st.session_state.results
    
    # Get the relevant data from the results
    messages = results.get("messages", [])
    
    # Find initial user stories, feedback, and final user stories
    initial_user_stories = None
    feedback = results.get("feedback", "No feedback provided.")
    final_user_stories = None
    
    if len(messages) >= 1:
        initial_user_stories = messages[0].content
    
    if results.get("product_decision") == "Feedback" and len(messages) >= 2:
        final_user_stories = messages[-3].content  # Adjust based on your workflow
    else:
        final_user_stories = initial_user_stories
    
    # User Stories Section
    st.header("User Stories")
    
    with st.expander("Initial User Stories", expanded=True):
        st.markdown(initial_user_stories)
    
    feedback_status = "✅ Approved" if results.get("product_decision") == "Approved" else "🔄 Needs Revision"
    with st.expander(f"Feedback ({feedback_status})"):
        st.markdown(feedback)
    
    if results.get("product_decision") == "Feedback":
        with st.expander("Revised User Stories", expanded=True):
            st.markdown(final_user_stories)
    
    # Documentation Section
    st.header("Documentation")
    
    # Technical Documentation
    tech_doc = results.get("technical_documentation", "")
    with st.expander("Technical Documentation", expanded=True):
        st.markdown(tech_doc)
    
    st.markdown(get_download_link(tech_doc, "technical_documentation.md"), unsafe_allow_html=True)
    
    # Functional Documentation
    func_doc = results.get("functional_documentation", "")
    with st.expander("Functional Documentation", expanded=True):
        st.markdown(func_doc)
    
    st.markdown(get_download_link(func_doc, "functional_documentation.md"), unsafe_allow_html=True)

else:
    # Display instructions
    st.info("👈 Fill in the project details in the sidebar and click 'Generate Documentation' to get started.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("What you'll get:")
        st.markdown("""
        - User stories based on your features
        - Expert review and feedback
        - Technical documentation
        - Functional documentation
        - Downloadable markdown files
        """)
    
    with col2:
        st.subheader("Tips for best results:")
        st.markdown("""
        - Provide a clear project description
        - List features one per line
        - Be specific about project requirements
        - Include user roles if possible
        """)