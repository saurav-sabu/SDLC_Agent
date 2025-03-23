import streamlit as st
import os
import tempfile
from typing import TypedDict, Annotated, Literal, Optional, List
from pydantic import Field, BaseModel
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langchain_core.messages import *
import time

st.set_page_config(
    page_title="LangGraph Development Assistant",
    page_icon="📝",
    layout="wide"
)

def load_css():
    """Add custom CSS to the app."""
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
        }
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 1rem;
        }
        .download-btn {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .download-btn:hover {
            background-color: #45a049;
        }
        .expander-header {
            font-size: 1.2em;
            font-weight: bold;
        }
        .step-complete {
            color: #4CAF50;
            font-weight: bold;
        }
        .step-in-progress {
            color: #FFA500;
            font-weight: bold;
        }
        .step-pending {
            color: #808080;
            font-style: italic;
        }
    </style>
    """, unsafe_allow_html=True)


def setup_api_key():
    """Setup API key from environment or user input."""
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    
    if not api_key:
        api_key = st.sidebar.text_input("Enter GOOGLE API Key:", type="password")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
    
    return api_key


def initialize_state():
    """Initialize session state variables."""
    if "project_name" not in st.session_state:
        st.session_state.project_name = ""
    if "project_description" not in st.session_state:
        st.session_state.project_description = ""
    if "features" not in st.session_state:
        st.session_state.features = []
    if "workflow_started" not in st.session_state:
        st.session_state.workflow_started = False
    if "current_step" not in st.session_state:
        st.session_state.current_step = ""
    if "user_stories" not in st.session_state:
        st.session_state.user_stories = ""
    if "product_feedback" not in st.session_state:
        st.session_state.product_feedback = ""
    if "revised_user_stories" not in st.session_state:
        st.session_state.revised_user_stories = ""
    if "technical_documentation" not in st.session_state:
        st.session_state.technical_documentation = ""
    if "functional_documentation" not in st.session_state:
        st.session_state.functional_documentation = ""
    if "combined_documentation" not in st.session_state:
        st.session_state.combined_documentation = ""
    if "design_feedback" not in st.session_state:
        st.session_state.design_feedback = ""
    if "generated_code" not in st.session_state:
        st.session_state.generated_code = ""
    if "code_files" not in st.session_state:
        st.session_state.code_files = {}
    
    if "workflow_complete" not in st.session_state:
        st.session_state.workflow_complete = False
    


def save_to_file(content, filename):
    """Save content to a temporary file and create a download link."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as tmp:
        tmp.write(content.encode())
        return tmp.name


def create_download_link(content, filename, button_text):
    """Create a download button for the given content."""
    temp_path = save_to_file(content, filename)
    with open(temp_path, 'rb') as file:
        st.download_button(
            label=button_text,
            data=file,
            file_name=filename,
            mime="text/markdown" if filename.endswith('.md') else "text/plain"
        )


def parse_code_blocks(code_content):
    """Parse code content to extract multiple code files."""
    code_files = {}
    current_file = None
    current_content = []
    lines = code_content.split('\n')
    
    for line in lines:
        if line.startswith("```") and "```" in line[:4]:
            # Check if we're closing a code block
            if current_file:
                code_files[current_file] = '\n'.join(current_content)
                current_file = None
                current_content = []
            # Check if we're opening a new code block with a filename
            elif len(line) > 3:
                lang_or_filename = line[3:].strip()
                # If it has an extension, treat as filename
                if "." in lang_or_filename and not lang_or_filename.startswith("json") and not lang_or_filename.startswith("xml"):
                    current_file = lang_or_filename
                    current_content = []
        elif current_file:
            current_content.append(line)
    
    # Handle case where no explicit filenames were found
    if not code_files and code_content:
        # Try to identify language-specific code blocks
        import re
        code_blocks = re.findall(r"```(\w+)(.*?)```", code_content, re.DOTALL)
        
        for idx, (lang, code) in enumerate(code_blocks):
            if lang in ["python", "py"]:
                code_files[f"main_{idx}.py"] = code.strip()
            elif lang in ["javascript", "js"]:
                code_files[f"script_{idx}.js"] = code.strip()
            elif lang in ["html"]:
                code_files[f"index_{idx}.html"] = code.strip()
            elif lang in ["css"]:
                code_files[f"style_{idx}.css"] = code.strip()
            else:
                code_files[f"file_{idx}.{lang}"] = code.strip()
    
    return code_files


# Define the State class for LangGraph
class State (TypedDict):
  messages: Annotated[list,add_messages]
  project_name: str
  project_description: str
  features: list[str]
  user_stories: str
  product_decision: str
  product_feedback: str
  final_product_feedback: str
  functional_documentation:str
  technical_documentation:str
  combined_documentation:str
  feedback_design:str
  design_decision:str
  generated_code:str
  code_decision:str
  code_feedback:str
  code_quality_score: str
  security_decision: str
  security_feedback: str
  security_review_response:str
  fix_security_response:str
  test_cases_decision:str
  test_cases_feedback:str
  test_cases_response:str
  qa_testing_decision:str
  qa_testing_feedback:str


# Define the structured output for product owner routing
class ProductOwnerRoute(BaseModel):
  step: Literal["Approved","Feedback"] = Field(description="The next step in the routing process")
  feedback: str = Field(description="If the user stories are not good, provide Feedback on how to improve them")


# Define the structured output for design routing
class DesignRoute(BaseModel):
    step: Literal["Approved", "Feedback"] = Field(description="The next step in routing process")
    feedback: str = Field(description="If the design documents are not good, provide feedback on how to improve them.")


def create_langgraph_workflow(api_key):
    """Create and return the LangGraph workflow."""
    if not api_key:
        st.error("Please provide an OpenAI API key to continue.")
        return None
    
    # Initialize LLM instance
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    router_product_owner_route = llm.with_structured_output(ProductOwnerRoute)
    router_design_route = llm.with_structured_output(DesignRoute)
  


    # Node Functions
    def generate_user_stories(state: State):
        st.session_state.current_step = "Generating User Stories"
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
        
        # Store in session state for UI
        st.session_state.user_stories = response.content

        return {"messages": messages}
    
    def generate_user_stories(state:State):
        st.session_state.current_step = "Generating User Stories"
        user_story_prompt = f"""
        You are an expert Agile product owner specializing in user story generation. Your goal is to create well-structured user stories
        that align with Agile best practices and ensure clarity for development teams. Each user story must adhere to the following:

        - Clearly define the user role.
        - Describe the action or feature the user needs.
        - Specify the benefit or reason behind the feature.
        - Include three to five well-defined acceptance criteria.
        - Ensure that the user stories are clear, concise, and aligned with Agile methodologies.
        - Consider the technical implementation constraints and opportunities.

        **Project Details:**
        - Project Name: {state["project_name"]}
        - Project Description: {state["project_description"]}
        - Key Features & Requirements:
        {state["features"]}


        **Stakeholder Feedback:**
        {state.get("product_feedback", "")}


        **Deliverables:**
        Generate at least one user story per key feature, ensuring:
        - The stories follow the standard format: "As a [user role], I want [feature or action] so that [benefit or reason]."
        - Each story has 3-5 acceptance criteria, formatted with bullet points.
        - If applicable, include edge cases or dependencies.
        - Suggest any technical constraints or opportunities the development team should be aware of.
        - Add complexity estimation (story points) if appropriate.
        - If there is any feedback consider it as well

        Please generate the structured user stories below:
        """

        response = llm.invoke([user_story_prompt] + state["messages"])
        st.session_state.user_stories = response.content
        return {"messages":response.content,"user_stories":response.content}
            
    
    def product_owner_review(state:State):
        st.session_state.current_step = "Product Owner Review"
        message_content = state["messages"][-1].content

        decision = router_product_owner_route.invoke(
            [
                SystemMessage(content="""Route the input to Approved or Feedback based on user stories quality.
                    If 'Approved', leave feedback empty or provide positive reinforcement.
                    If 'Feedback', provide constructive feedback on how to improve the stories."""),
                    HumanMessage(content=message_content),
            ]
        )
        print(f"Decision Step: {decision.step}")
        print(f"Feedback: {decision.feedback}")
        st.session_state.product_feedback = decision.feedback
        return {"product_decision":decision.step,"product_feedback":decision.feedback}
    
    def route_product_decision(state:State):
        """Routes the workflow based on product owner decision."""
        if state["product_decision"] == "Feedback":
            return "Feedback"
        else:
            return "Approved"
    
    def revise_user_stories(state:State):
        st.session_state.current_step = "Revising User Stories"
        revise_prompt = f"""
        You are an expert Agile product owner refining user stories based on specific feedback. Your goal is to transform these stories into production-ready specifications that development teams can implement with minimal clarification needed.

        **Project Context:**
        - Project Name: {state["project_name"]}
        - Project Description: {state["project_description"]}
        - Key Features:
        {state["features"]}

        **Stakeholder Feedback:**
        {state["product_feedback"]}

        **Original User Stories:**
        {state["user_stories"]}

        **Your Task:**
        Transform these user stories by:
        1. Ensuring perfect adherence to the format: "As a [specific user role], I want [concrete feature or action] so that [tangible benefit or business value]"
        2. Providing exactly 3-5 testable acceptance criteria for each story using the Given/When/Then format
        3. Eliminating all ambiguities and subjective language (e.g., "user-friendly," "intuitive")
        4. Identifying dependencies between stories and noting them explicitly
        5. Addressing edge cases and error conditions
        6. Breaking down stories that are too large into smaller, independently valuable chunks

        **Expected Output Format:**
        For each user story, provide:
        - The refined user story statement
        - 3-5 acceptance criteria in Given/When/Then format
        - Dependencies (if applicable)
        - Story points estimate (if applicable)

        Return the refined user stories in dictionary format to maintain compatibility with the graph structure.
        """
        revised_response = llm.invoke([revise_prompt] + state["messages"])
        st.session_state.revised_user_stories = revised_response.content
        return {"messages":revised_response.content,"final_product_feedback":revised_response.content}


    def generate_technical_documents(state:State):
        st.session_state.current_step = "Generating Technical Documentation"
        technical_documentation_prompt = f"""
        ## **Technical Documentation Generation Prompt**

        You are an expert **technical writer and software architect** specializing in software documentation. Your task is to generate a **detailed, structured, and well-organized technical documentation** based on the provided **Functional Specification Document (FSD)** and **User Stories**. Ensure clarity, completeness, and adherence to **industry best practices**.

        ---

        ## **Inputs**
        1. **User Stories**
        {state["user_stories"]}

        2. **Project Details**
        - **Project Name**: {state["project_name"]}
        - **Project Description**: {state["project_description"]}
        - **Features**: {state["features"]}

        ## **Documentation Structure**
        The technical documentation should be structured as follows:

        ### **1. Introduction**
        - Provide an overview of the system, including its **purpose, key objectives, and business goals**.
        - Define the **target audience, stakeholders, and primary users**.
        - Summarize how the system addresses business needs.

        ### **2. System Architecture**
        - Describe the **high-level architecture** using diagrams where necessary.
        - List the **technologies, frameworks, and programming languages** used.
        - Explain the interaction between **backend, frontend, database, APIs, and external services**.
        - Highlight **scalability considerations and system constraints**.

        ### **3. Features & Functionalities**
        For each user story, document:
        - **Feature Name**:
        - **User Story** ("As a [user], I want [feature], so that [benefit]"):
        - **Functional Requirement** (as detailed in the FSD):
        - **Acceptance Criteria** (clear and testable conditions for completion):
        - **System Behavior** (expected inputs, outputs, and interactions):
        - **Edge Cases & Error Handling** (uncommon but critical scenarios):

        ### **4. API Documentation**
        For each API endpoint, provide:
        - **Endpoint URL**:
        - **HTTP Method** (GET, POST, PUT, DELETE):
        - **Request Parameters** (headers, query parameters, body format):
        - **Response Format** (success and error responses with examples):
        - **Authentication Mechanism** (OAuth, JWT, API keys):
        - **Error Handling & Status Codes**:

        ### **5. Database Schema**
        - Provide an **Entity Relationship Diagram (ERD)** or a structured description of database tables.
        - Define **primary keys, foreign keys, indexes, and constraints**.
        - Explain how data is stored, retrieved, and related to system functionality.

        ### **6. Security Considerations**
        - Define **authentication and authorization mechanisms**.
        - Discuss **data encryption methods** for data at rest and in transit.
        - Highlight **security policies, regulatory compliance, and best practices**.

        ### **7. Performance & Scalability**
        - Mention **caching strategies, load balancing, and rate limiting**.
        - Describe system behavior under **high load conditions**.
        - Identify potential **bottlenecks and optimization techniques**.

        ### **8. Deployment & DevOps Strategy**
        - Outline the **CI/CD pipeline**, including build, test, and deployment processes.
        - Describe **cloud infrastructure, containerization (Docker, Kubernetes), and orchestration tools**.
        - Explain different **environments (development, staging, production)**.

        ### **9. Testing Strategy**
        - Detail the **testing methodologies**, including:
        - **Unit Testing** (isolated component testing)
        - **Integration Testing** (testing interaction between components)
        - **Performance Testing** (stress, load, and scalability tests)
        - **Security Testing** (penetration testing, vulnerability assessment)

        ### **10. Maintenance & Future Enhancements**
        - Define **logging and monitoring** strategies.
        - Outline **error handling, alerts, and system observability tools**.
        - Provide a roadmap for **future improvements and scalability planning**.

        ---

        ## **Output Format**
        - Generate the documentation in **Markdown format** with structured headings, bullet points, and code blocks.
        - Ensure clarity, completeness, and readability.
        - Adhere to **technical writing best practices** and **industry-standard conventions**.
        """


        technical_response = llm.invoke([technical_documentation_prompt] + state["messages"])
        print("Technical Response:")
        st.session_state.technical_documentation = technical_response.content
        return {"messages":technical_response.content,"technical_documentation":technical_response.content}

    
    def generate_functional_documents(state:State):
        st.session_state.current_step = "Generating Functional Documentation"
        functional_documentation_prompt = f"""
        ## **Functional Specification Document (FSD) Prompt**

        You are an expert technical writer specializing in software documentation. Your task is to generate a **comprehensive and structured Functional Specification Document (FSD)** for a project based on the provided user stories. Ensure that the document is **clear, concise, and aligned with industry best practices**.

        ---

        ## **Project Overview**
        - **Project Name**: {state["project_name"]}
        - **Project Description**: {state["project_description"]}
        - **Features**: {state["features"]}

        ## **Functional Requirements**
        Define the functional requirements in detail for each user story.
        {state["user_stories"]}

        ---
        ## **Functional Specification Format**
        The document should be structured as follows:

        ### **1. Introduction**
        - Provide an overview of the project and its objectives.
        - Explain the business needs and the problem it addresses.
        - Identify key stakeholders and target end-users.

        ### **2. Scope of the System**
        - Clearly define what is **in scope** and **out of scope** for the system.
        - List assumptions and dependencies that impact the system’s functionality.

        ### **3. System Features & Functionalities**
        For each feature derived from user stories, include:
        - **Feature Name**:
        - **User Story** (format: "As a [user], I want [feature], so that [benefit]"):
        - **Functional Requirement** (detailed breakdown of expected behavior):
        - **Preconditions** (conditions that must be met before execution):
        - **Main Flow** (step-by-step sequence of events in normal use case):
        - **Alternate Flows & Edge Cases** (scenarios deviating from the main flow):
        - **Postconditions** (expected state after successful execution):

        ### **4. User Interface (UI) Specifications**
        - Describe key UI components, user interactions, and navigation flow.
        - Provide wireframes or mockups (if available).
        - Mention accessibility considerations and UI responsiveness.

        ### **5. Data Flow & Processing**
        - Explain how data moves through the system and how it's processed.
        - Define key data transformations, validation rules, and storage mechanisms.

        ### **6. Integration Points**
        - List external systems, APIs, or databases the system interacts with.
        - Define data exchange formats (JSON, XML, etc.) and integration protocols (REST, GraphQL, WebSockets).
        - Specify API authentication and authorization mechanisms.

        ### **7. Security & Compliance**
        - Define authentication, authorization, and access control policies.
        - Mention encryption standards for data at rest and in transit.
        - Address regulatory compliance (e.g., GDPR, HIPAA, ISO 27001).

        ### **8. Performance & Scalability Considerations**
        - Specify system response times, concurrency limits, and expected throughput.
        - Discuss caching strategies, load balancing, and scaling approaches.

        ### **9. Error Handling & Logging**
        - Describe error handling strategies and fallback mechanisms.
        - Define structured logging formats and log retention policies.
        - Include failure recovery strategies (e.g., retry mechanisms, alerts).

        ### **10. Constraints & Limitations**
        - Define hardware/software limitations, licensing constraints, or technology dependencies.

        ### **11. Acceptance Criteria & Validation**
        - Define the criteria for determining feature completion and acceptance.
        - Outline functional test cases, expected outcomes, and validation steps.

        ---

        ## **Output Format**
        - Generate the document in **Markdown format** with well-structured headings, bullet points, and code snippets (where applicable).
        - Ensure **clarity, completeness, and adherence to best practices**.
        - Maintain **technical accuracy and consistency** throughout the document.
        """

        functional_response = llm.invoke([functional_documentation_prompt] + state["messages"])
        print("Functional Response:")
        st.session_state.functional_documentation = functional_response.content
        return {"messages":functional_response.content,"functional_documentation":functional_response.content}

    
    def generate_combined_documentation(state: State):
        st.session_state.current_step = "Generating Combined Documentation"
        feedback_design = state.get("feedback_design", "")
        improved_combine_doc = f"""
#   Comprehensive Project Documentation

**Project Name**: {state["project_name"]}  
**Project Description**: {state["project_description"]}

## Functional Documentation
{state['functional_documentation']}

## Technical Documentation
{state['technical_documentation']}

## Feedback from Design Review
{feedback_design}

**Note:** This document serves as a **single source of truth** for both functional and technical teams. It ensures that business requirements are accurately translated into technical implementations.
"""
        combine_message = llm.invoke([improved_combine_doc] + state["messages"])
        print(combine_message)
        st.session_state.combined_documentation = combine_message.content
        return {"messages":combine_message.content,"combined_documentation": combine_message.content}

    
    def design_review(state: State):
        st.session_state.current_step = "Design Review"
        
        """Routes the user stories for approval or revision."""
        message_content = state["combined_documentation"]  # Extract content from last message
        decision = router_design_route.invoke(
            [
                SystemMessage(content="""Route the input to Approved or Feedback based on technical and functional document quality.
                If 'Approved', leave feedback empty or provide positive reinforcement.
                If 'Feedback', provide constructive feedback on how to improve the technical and functional document quality."""),
                HumanMessage(content=message_content),
            ]
        )
        
        # Store in session state for UI
        st.session_state.design_feedback = decision.feedback
        
        return {"design_decision": decision.step, "feedback": decision.feedback}
    
    def route_design_decision(state: State):
        """Routes the workflow based on product owner decision."""
        if state["design_decision"] == "Feedback":
            return "Feedback"
        else:  # "Approved"
            return "Approved"
    
    def generate_code_from_documentation(state: State):
        st.session_state.current_step = "Generating Code"
        
        code_generation_prompt = f"""
🔹 **Software Implementation Request** 🔹

You are a highly skilled **software engineer** specializing in **building scalable, maintainable, and secure applications**. Your task is to generate **implementation-ready code** based on the following **comprehensive documentation**.

---

## **📌 Project Overview**
- **Project Name**: {state["project_name"]}
- **Project Description**: {state["project_description"]}

---

## **📖 Refined Technical & Functional Documentation**
{state["combined_documentation"]}

## **Code Feedback**
{state.get("code_quality_score", "")}

## **QA Testing Feedback**
{state.get("qa_testing_feedback", "")}
---

## **🛠️ Implementation Guidelines**
🔹 **Core Features & Functionalities**
- Implement all key features as described in the documentation.
- Ensure that business logic is correctly translated into code.

🔹 **Software Architecture & Best Practices**
- Follow modular and scalable **code architecture**.
- Ensure **separation of concerns (SoC)** for better maintainability.
- Implement **design patterns** where applicable (MVC, Repository, etc.).
- Use **efficient data structures and algorithms** where needed.

🔹 **API Development**
- Define **RESTful API endpoints** or **GraphQL schemas** (as per documentation).
- Implement CRUD operations and necessary **authentication & authorization**.
- Follow **proper API versioning and documentation** (e.g., OpenAPI/Swagger).

🔹 **Database & Storage**
- Define **optimized database schemas** (SQL/NoSQL as per requirements).
- Ensure **indexing, normalization, and query optimization**.
- Implement **data validation and integrity constraints**.

🔹 **Security & Error Handling**
- Use **robust error handling mechanisms** (try-except, proper logging).
- Implement **input validation & sanitization** to prevent security vulnerabilities (SQL injection, XSS, CSRF, etc.).
- Ensure **secure authentication & authorization** (JWT, OAuth, etc.).
- Follow **secure coding principles** to mitigate common threats.

🔹 **Performance & Scalability**
- Optimize for **low-latency API responses**.
- Implement **caching mechanisms** (Redis, Memcached) for performance.
- Ensure **asynchronous processing** where needed (Celery, AsyncIO, Kafka).

🔹 **Code Readability & Documentation**
- Provide **meaningful comments and docstrings**.
- Maintain **consistent code formatting and naming conventions**.
- Structure code for **readability and ease of collaboration**.

---

## **📌 Expected Output**
- **Fully structured implementation-ready code**.
- Well-organized modules and functions.
- Code snippets in **appropriate files & folders**.
- Necessary configurations, environment variables, and setup instructions.

🚀 **Deliver the code in a well-structured format with explanations where necessary.**
"""

        code_response = llm.invoke([code_generation_prompt] + state["messages"])

        generated_code = code_response.content  # Store the generated code separately
        
        # Parse code files and store in session state
        code_files = parse_code_blocks(generated_code)
        st.session_state.code_files = code_files
        st.session_state.generated_code = generated_code
        st.session_state.workflow_complete = True

        return {"messages": code_response.content, "generated_code": code_response.content}
    
    
    # Create the graph
    builder = StateGraph(State)

    # Add nodes
    builder.add_node("Auto Generate User Stories", generate_user_stories)
    builder.add_node("Product Owner Review", product_owner_review)
    builder.add_node("Revise User Stories", revise_user_stories)
    builder.add_node("Generate Functional Documentation", generate_functional_documents)
    builder.add_node("Generate Technical Documentation", generate_technical_documents)
    builder.add_node("Generate Combined Documentation", generate_combined_documentation)
    builder.add_node("Design Review", design_review)
    builder.add_node("Generate Code", generate_code_from_documentation)

    # Add edges
    builder.add_edge(START, "Auto Generate User Stories")
    builder.add_edge("Auto Generate User Stories","Product Owner Review")

    builder.add_conditional_edges(
        "Product Owner Review",
        route_product_decision,
        {
            "Approved": "Generate Functional Documentation",
            "Feedback": "Revise User Stories"
        }
    )


    builder.add_edge("Revise User Stories","Auto Generate User Stories")
    builder.add_edge("Product Owner Review", "Generate Technical Documentation")
    builder.add_edge("Generate Technical Documentation", "Generate Combined Documentation")
    builder.add_edge("Generate Functional Documentation", "Generate Combined Documentation")
    builder.add_edge("Generate Combined Documentation", "Design Review")

    builder.add_conditional_edges(
        "Design Review",
        route_design_decision,
        {
            "Approved": "Generate Code",
            "Feedback": "Generate Combined Documentation"
        }
    )

    builder.add_edge("Generate Code", END)

    graph = builder.compile()
    return graph


def display_progress_tracker():
    """Display progress tracker for workflow steps."""
    st.sidebar.markdown("### Workflow Progress")
    
    steps = [
        "Generate User Stories",
        "Product Owner Review",
        "Revise User Stories",
        "Generate Technical Documentation",
        "Generate Functional Documentation",
        "Generate Combined Documentation",
        "Design Review",
        "Generate Code"
    ]
    
    current_step = st.session_state.current_step
    
    for step in steps:
        if step in current_step:
            st.sidebar.markdown(f"- <span class='step-in-progress'>⏳ {step}</span>", unsafe_allow_html=True)
        elif any(s in st.session_state.current_step for s in steps[steps.index(step):]) or not st.session_state.workflow_started:
            st.sidebar.markdown(f"- <span class='step-pending'>🔄 {step}</span>", unsafe_allow_html=True)
        else:
            st.sidebar.markdown(f"- <span class='step-complete'>✅ {step}</span>", unsafe_allow_html=True)


def main():
    """Main function to run the Streamlit app."""
    load_css()
    initialize_state()
    api_key = setup_api_key()
    
    st.title("🚀 LangGraph Development Assistant")
    st.markdown("Generate user stories, technical documentation, and implementation code from project details.")
    
    # Display progress tracker in sidebar
    display_progress_tracker()
    
    # Project Details Input
    st.header("Project Details")
    
    with st.form("project_details_form"):
        project_name = st.text_input("Project Name", st.session_state.project_name)
        project_description = st.text_area("Project Description", st.session_state.project_description, height=100)
        
        features_input = st.text_area(
            "Features (One per line)",
            "\n".join(st.session_state.features) if st.session_state.features else "",
            height=150,
            placeholder="Login Authentication\nUser Profile Management\nSearch Functionality"
        )
        
        submit_button = st.form_submit_button("Start Development Workflow")
        
        if submit_button:
            if not project_name or not project_description or not features_input:
                st.error("Please fill in all fields to continue.")
            else:
                st.session_state.project_name = project_name
                st.session_state.project_description = project_description
                st.session_state.features = [f for f in features_input.split("\n") if f.strip()]
                st.session_state.workflow_started = True
    
    # Start workflow if all inputs are provided
    if st.session_state.workflow_started:
        workflow = create_langgraph_workflow(api_key)
        
        if workflow and not st.session_state.workflow_complete:
            with st.spinner("Running development workflow... This may take a few minutes..."):
                # Create inputs dictionary for workflow
                inputs = {
                    "project_name": st.session_state.project_name,
                    "project_description": st.session_state.project_description,
                    "features": st.session_state.features,
                    "messages": []
                }
                
                # Run the workflow
                for event in workflow.stream(inputs):
                    # Debug output if needed
                    # st.write(event)
                    pass
                
                st.success("Workflow completed successfully!")
                st.session_state.workflow_complete = True
                
        # Display results in tabs
        if st.session_state.user_stories:
            tabs = st.tabs(["User Stories", "Documentation", "Generated Code"])
            
            # User Stories Tab
            with tabs[0]:
                st.header("User Story Generation")
                
                with st.expander("📝 Initial User Stories", expanded=True):
                    st.markdown(st.session_state.user_stories)
                    create_download_link(
                        st.session_state.user_stories,
                        "initial_user_stories.md",
                        "Download Initial User Stories"
                    )
                
                if st.session_state.product_feedback:
                    with st.expander("💬 Product Owner Feedback", expanded=True):
                        st.info(st.session_state.product_feedback)
                
                if st.session_state.revised_user_stories:
                    with st.expander("✅ Revised User Stories", expanded=True):
                        st.markdown(st.session_state.revised_user_stories)
                        create_download_link(
                            st.session_state.revised_user_stories,
                            "revised_user_stories.md",
                            "Download Revised User Stories"
                        )
            
            # Documentation Tab
            with tabs[1]:
                st.header("Project Documentation")
                
                if st.session_state.technical_documentation:
                    with st.expander("📊 Technical Documentation", expanded=True):
                        st.markdown(st.session_state.technical_documentation)
                        create_download_link(
                            st.session_state.technical_documentation,
                            f"{st.session_state.project_name.lower().replace(' ', '_')}_technical_documentation.md",
                            "Download Technical Documentation"
                        )
                
                if st.session_state.functional_documentation:
                    with st.expander("📋 Functional Documentation", expanded=True):
                        st.markdown(st.session_state.functional_documentation)
                        create_download_link(
                            st.session_state.functional_documentation,
                            f"{st.session_state.project_name.lower().replace(' ', '_')}_functional_documentation.md",
                            "Download Functional Documentation"
                        )
                
                if st.session_state.combined_documentation:
                    with st.expander("📚 Combined Documentation", expanded=True):
                        st.markdown(st.session_state.combined_documentation)
                        create_download_link(
                            st.session_state.combined_documentation,
                            f"{st.session_state.project_name.lower().replace(' ', '_')}_combined_documentation.md",
                            "Download Combined Documentation"
                        )
                
                if st.session_state.design_feedback:
                    with st.expander("💬 Design Review Feedback", expanded=False):
                        st.info(st.session_state.design_feedback)
            
            # Generated Code Tab
            with tabs[2]:
                st.header("Implementation Code")
                
                if st.session_state.code_files:
                    st.write(f"Generated {len(st.session_state.code_files)} code files:")
                    
                    # Display each code file in an expander
                    for filename, content in st.session_state.code_files.items():
                        with st.expander(f"📄 {filename}", expanded=False):
                            st.code(content, language=filename.split('.')[-1])
                            create_download_link(
                                content,
                                filename,
                                f"Download {filename}"
                            )
                    
                    # Option to download all files as a zip
                    if st.session_state.code_files:
                        import zipfile
                        import io
                        
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            for filename, content in st.session_state.code_files.items():
                                zip_file.writestr(filename, content)
                        
                        st.download_button(
                            label="Download All Code Files (ZIP)",
                            data=zip_buffer.getvalue(),
                            file_name=f"{st.session_state.project_name.lower().replace(' ', '_')}_code.zip",
                            mime="application/zip"
                        )
                elif st.session_state.generated_code:
                    st.markdown("### Raw Generated Code")
                    st.code(st.session_state.generated_code)
                    create_download_link(
                        st.session_state.generated_code,
                        "generated_code.md",
                        "Download Generated Code"
                    )


if __name__ == "__main__":
    main()