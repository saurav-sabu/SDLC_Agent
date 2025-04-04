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
    if "code_quality_score" not in st.session_state:
        st.session_state.code_quality_score = ""
    if "code_feedback" not in st.session_state:
        st.session_state.code_feedback = ""
    if "fixed_code_after_code_review" not in st.session_state:
        st.session_state.fixed_code_after_code_review = ""
    if "fixed_code_after_security" not in st.session_state:
        st.session_state.fixed_code_after_security = ""
    if "fixed_code_after_qa_feedback" not in st.session_state:
        st.session_state.fixed_code_after_qa_feedback = ""
    if "fixed_test_cases_after_review" not in st.session_state:
        st.session_state.fixed_test_cases_after_review = ""
    if "security_feedback" not in st.session_state:
        st.session_state.security_feedback = ""
    if "security_decision" not in st.session_state:
        st.session_state.security_decision = ""
    if "test_cases_feedback" not in st.session_state:
        st.session_state.test_cases_feedback = ""
    if "test_cases_decision" not in st.session_state:
        st.session_state.test_cases_decision = ""
    if "test_cases_response" not in st.session_state:
        st.session_state.test_cases_response = ""
    if "qa_testing_feedback" not in st.session_state:
        st.session_state.qa_testing_feedback = ""
    if "qa_testing_decision" not in st.session_state:
        st.session_state.qa_testing_decision = ""
    if "write_test_cases_response" not in st.session_state:
        st.session_state.write_test_cases_response = ""
    if "qa_final_feedback" not in st.session_state:
        st.session_state.qa_final_feedback = ""
    if "security_review_response" not in st.session_state:
        st.session_state.security_review_response = ""

    
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
  write_test_cases_response: str
  qa_testing_decision:str
  qa_testing_feedback:str
  qa_final_feedback:str


# Define the structured output for product owner routing
class ProductOwnerRoute(BaseModel):
  step: Literal["Approved","Feedback"] = Field(description="The next step in the routing process")
  feedback: str = Field(description="If the user stories are not good, provide Feedback on how to improve them")


# Define the structured output for design routing
class DesignRoute(BaseModel):
    step: Literal["Approved", "Feedback"] = Field(description="The next step in routing process")
    feedback: str = Field(description='If the design documents are not good, provide feedback on how to improve them. If good, leave "".')

class CodeReviewRoute(BaseModel):
    step: Literal["Approved", "Feedback"] = Field(
        description="The next step in routing process after code review"
    )
    feedback: str = Field(
        description="If the code is not approved, provide feedback on how to improve it."
    )

class SecurityReviewRoute(BaseModel):
    step: Literal["Approved", "Feedback"] = Field(
        description="The next step in routing process after security review"
    )
    feedback: str = Field(
        description="If security issues are found, provide feedback on how to fix them."
    )

class TestCasesReviewRoute(BaseModel):
    step: Literal["Approved", "Feedback"] = Field(
        description="The next step in routing process after test cases review"
    )
    feedback: str = Field(
        description="If the test cases are not approved, provide feedback on how to improve them."
    )



def create_langgraph_workflow(api_key):
    """Create and return the LangGraph workflow."""
    if not api_key:
        st.error("Please provide an OpenAI API key to continue.")
        return None
    
    # Initialize LLM instance
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    router_product_owner_route = llm.with_structured_output(ProductOwnerRoute)
    router_design_route = llm.with_structured_output(DesignRoute)
    router_code_review_route = llm.with_structured_output(CodeReviewRoute)
    router_security_review_route = llm.with_structured_output(SecurityReviewRoute)
    router_test_cases_review_route = llm.with_structured_output(TestCasesReviewRoute)


    
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
        {state.get("final_product_feedback", "")}

        **Deliverables:**
        Generate at least one user story per key feature, ensuring:
        - The stories follow the standard format: "As a [user role], I want [feature or action] so that [benefit or reason]."
        - Each story has 3-5 acceptance criteria, formatted with bullet points.
        - If applicable, include edge cases or dependencies.
        - Suggest any technical constraints or opportunities the development team should be aware of.
        - Add complexity estimation (story points) if appropriate.
        - If there is any feedback from stakeholder consider it as well.

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
        You are an expert Agile product owner evaluating and refining user stories based on stakeholder feedback. Your goal is to analyze the provided user stories and suggest general areas for improvement to ensure they are production-ready and align with Agile best practices.

        ### **Project Context:**
        - **Project Name:** {state["project_name"]}
        - **Project Description:** {state["project_description"]}
        - **Key Features:**  
        {state["features"]}

        ### **Stakeholder Feedback:**
        {state["product_feedback"]}

        ### **Original User Stories:**
        {state["user_stories"]}

### **Your Task:**
Review the provided user stories and offer **general suggestions for improvement** that can be applied across all stories. Your suggestions should focus on:

1. **Clarity & Format:** Are the user stories structured in the correct format? Do they clearly define the user role, action, and benefit?  
2. **Acceptance Criteria:** Are the acceptance criteria well-defined, testable, and following the Given/When/Then format?  
3. **Ambiguity & Subjectivity:** Are there any vague terms that should be replaced with more precise descriptions?  
4. **Story Breakdown & Dependencies:** Are any user stories too broad and in need of breaking down? Are dependencies between stories clearly identified?  
5. **Edge Cases & Error Handling:** Are important error conditions and alternative flows considered?  
6. **Feasibility & Testability:** Can each story be implemented within a sprint? Is it independently valuable and testable?  

### **Expected Output:**  
Provide **general recommendations** rather than rewriting individual user stories. Your response should highlight common issues, patterns, and best practices that can be applied across all user stories.  

Return your feedback in a **concise and actionable** format, ensuring it is applicable across multiple stories without listing each one separately.
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
# Comprehensive Project Documentation  

**Project Name**: {state["project_name"]}  
**Project Description**: {state["project_description"]}  

## Functional Documentation  
{state['functional_documentation']}  

## Technical Documentation  
{state['technical_documentation']}  

## Feedback from Design Review  
{feedback_design}  

---

### 🔹 **Important Notes:**  
- This document is formatted in **Markdown** to ensure clarity and readability.  
- It serves as a **single source of truth** for both functional and technical teams.  
- The structure ensures business requirements are accurately translated into technical implementations.  

Ensure that the output **strictly follows Markdown syntax** for proper rendering.
"""

        combine_message = llm.invoke([improved_combine_doc] + state["messages"])
        st.session_state.combined_documentation = combine_message.content
        return {"messages":combine_message.content,"combined_documentation": combine_message.content}

    
    def design_review(state: State):
        st.session_state.current_step = "Design Review"
        """Routes the user stories for approval or revision."""
        message_content = state["combined_documentation"]  # Extract content from last message
        decision = router_design_route.invoke(
            [
                SystemMessage(content="""Route the input to Approved or Feedback based on technical and functional document quality.
                If 'Approved', leave feedback as "" or provide positive reinforcement.
                If 'Feedback', provide constructive feedback on how to improve the technical and functional document quality."""),
                HumanMessage(content=message_content),
            ]
        )
        
        # Store in session state for UI
        
        st.session_state.design_feedback = decision.feedback
        
        return {"design_decision": decision.step, "feedback_design": decision.feedback}
    
    def route_design_decision(state: State):
        """Routes the workflow based on product owner decision."""
        if state["design_decision"] == "Feedback":
            return "Feedback"
        else:  # "Approved"
            return "Approved"
        
    def code_review(state:State):
        st.session_state.current_step = "Code Review"
        """Routes the user stories for approval or revision."""
        message_content = state["generated_code"]  # Extract content from last message
        # Use router_code_review_route instead of CodeReviewRoute
        decision = router_code_review_route.invoke(
                [
                    SystemMessage(content="""Route the code to Approved or Feedback based on quality.
                    If 'Approved', you can still provide minor suggestions for improvement.
                    If 'Feedback', provide detailed feedback on critical issues that must be fixed."""),
                    HumanMessage(content=message_content)
                ]
            )

        print(f"Decision Step: {decision.step}")
        print(f"Feedback: {decision.feedback}")
        return {"code_decision": decision.step, "code_feedback": decision.feedback}
    
    def route_code_review_decision(state: State):
        """Routes the workflow based on code review decision."""
        if state["code_decision"] == "Feedback":
            return "Feedback"
        else:  # "Approved"
            return "Approved"  # Move to the next step in your workflow
        
    
    def security_review(state:State):
        st.session_state.current_step = "Security Review"
        """Routes the user stories for approval or revision."""
        message_content = state["generated_code"]  # Extract content from last message
        # Use router_code_review_route instead of CodeReviewRoute
        decision = router_security_review_route.invoke(
                [
                    SystemMessage(content="""Route the code to Approved or Feedback based on security issues.
                    If 'Approved', you can still provide minor suggestions for improvement.
                    If 'Feedback', provide detailed feedback on critical security issues that must be fixed."""),
                    HumanMessage(content=message_content)
                ]
            )

        print(f"Decision Step: {decision.step}")
        print(f"Feedback: {decision.feedback}")
        st.session_state.security_feedback = decision.feedback
        return {"security_decision": decision.step, "security_feedback": decision.feedback}
    

    def route_security_review_decision(state: State):
        """Routes the workflow based on security review decision."""
        if state["security_decision"] == "Feedback":
            return "Feedback"
        else:  # "Approved"
            return "Approved"  # Move to the next step in your workflow
        
    def test_cases_review(state: State):
        st.session_state.current_step = "Test Cases Review"
        """Reviews test cases for approval or revision."""
        # Construct a prompt to review the test cases
        test_review_prompt = f"""
        You are an expert QA reviewer tasked with evaluating test cases for completeness and quality.

        **Project Name**: {state["project_name"]}
        **Project Description**: {state["project_description"]}

        **Test Cases to Evaluate**:
        {state["write_test_cases_response"]}

        **Review Criteria**:
        - Test coverage: do the tests cover all functions, modules, and features?
        - Edge cases: are boundary conditions and exceptional scenarios tested?
        - Test clarity: are the tests clearly written and well-documented?
        - Maintainability: can the tests be easily maintained as the code evolves?
        - Integration testing: are component interactions properly tested?
        - Error handling: are exception paths and error conditions tested?

        Evaluate if these test cases meet quality standards for production code.
        """

        # Use the structured output router
        decision = router_test_cases_review_route.invoke(
            [
                SystemMessage(content="""Review the test cases and determine if they provide adequate coverage.
                If coverage is below 80% or missing critical test scenarios, the decision must be 'Feedback'."""),
                HumanMessage(content=test_review_prompt),
            ]
        )
        print(f"Decision Step: {decision.step}")
        print(f"Feedback: {decision.feedback}")
        st.session_state.test_cases_feedback = decision.feedback
        return {"test_cases_decision": decision.step, "test_cases_feedback": decision.feedback}

    def route_test_cases_decision(state: State):
        """Routes the workflow based on test cases decision."""
        if state["test_cases_decision"] == "Feedback":
            return "Feedback"
        else:  # "Approved"
            return "Approved"
        
    def qa_testing(state: State):
        st.session_state.current_step = "QA Testing"
        """Performs QA testing on the code and determines if it passes or fails."""
        qa_testing_prompt = f"""
        You are an expert QA tester tasked with executing test cases and reporting results.

        **Project Name**: {state["project_name"]}
        **Project Description**: {state["project_description"]}

        **Code to Test**:
        {state["generated_code"]}

        **Test Cases**:
        {state["write_test_cases_response"]}

        **Testing Criteria**:
        - Functionality: Does the code perform as expected?
        - Reliability: Does it handle edge cases and errors gracefully?
        - Performance: Does it meet performance expectations?
        - Usability: Is the API/interface intuitive and consistent?
        - Security: Does it maintain security standards?

        **Testing Process**:
        1. Execute each test case
        2. Document the results (pass/fail)
        3. Record any unexpected behaviors or errors
        4. Measure performance metrics where applicable
        5. Provide an overall assessment

        Based on the code and test cases, simulate a thorough QA testing process and report your findings.
        """

        class QATestingResult(BaseModel):
            decision: Literal["Passed", "Failed"] = Field(
                description="Final testing decision - Passed or Failed"
            )
            feedback: str = Field(
                description="Detailed feedback including test results, issues found, and recommendations for improvement"
            )

        router_qa_testing = llm.with_structured_output(QATestingResult)

        test_results = router_qa_testing.invoke(
            [
                SystemMessage(content="""Execute a thorough QA testing simulation and provide detailed results.
                If any critical tests fail or if overall pass rate is below 90%, the decision must be 'Failed'."""),
                HumanMessage(content=qa_testing_prompt),
            ]
        )

        print(f"QA Testing Decision: {test_results.decision}")
        print(f"QA Testing Feedback: {test_results.feedback}")

        st.session_state.qa_testing_feedback = test_results.feedback

        return {
            "qa_testing_decision": test_results.decision,
            "qa_testing_feedback": test_results.feedback
        }

    def route_qa_testing_decision(state: State):
        """Routes the workflow based on QA testing results."""
        if state["qa_testing_decision"] == "Failed":
            return "Failed"
        else:  # "Passed"
            return "Passed"

    
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

## **Security Review Feedback**
{state.get("security_review_response", "")}

## **QA Testing Feedback**
{state.get("qa_final_feedback", "")}
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
- Necessary configurations, environment variables, and setup instructions in a README file.

🚀 **Deliver the code in a well-structured format.**
"""

        code_response = llm.invoke([code_generation_prompt] + state["messages"])

        generated_code = code_response.content  # Store the generated code separately

        print(generated_code)
        
        # Parse code files and store in session state
        code_files = parse_code_blocks(generated_code)
        st.session_state.code_files = code_files
        st.session_state.generated_code = generated_code
        st.session_state.workflow_complete = True

        return {"messages": code_response.content, "generated_code": code_response.content}
    
    def fix_code_after_code_review(state:State):
        st.session_state.current_step = "Fix Code After Code Review"
        code_review_prompt = f"""
        🔍 **Comprehensive Code Review Request** 🔍

        You are an **expert software engineer and code reviewer** with deep expertise in **clean code, performance optimization, and security best practices**.
        Your task is to **critically evaluate** the following code and provide **a detailed, structured review**.

        ---

        ## **📌 Project Overview**
        - **Project Name**: {state["project_name"]}
        - **Project Description**: {state["project_description"]}

        ---

        ## **📖 Generated Code for Review**
        {state["generated_code"]}

        ## **Feedback for code**
        {state.get("code_feedback", "")}

        ---

        ## **🛠️ Code Review Guidelines**
        Please review the code against the following **critical areas** and provide detailed feedback:

        ### ✅ **1. Code Quality & Readability**
        - Is the code **clean, well-structured, and modular**?
        - Are **function and variable names** meaningful and self-explanatory?
        - Are there **sufficient comments and docstrings** where needed?

        ### 🐛 **2. Bug Detection & Logic Errors**
        - Identify any **logical errors or unexpected behaviors**.
        - Check for **incorrect assumptions, undefined variables, or faulty conditions**.

        ### 🔒 **3. Security Best Practices**
        - Are there **potential security vulnerabilities** (e.g., SQL injection, XSS, CSRF, etc.)?
        - Is **authentication & authorization** implemented securely?
        - Are **sensitive data handling & encryption** properly managed?

        ### 🚀 **4. Performance & Optimization**
        - Are there any **performance bottlenecks**?
        - Can the code be optimized using **better algorithms or data structures**?
        - Are there **unnecessary computations, redundant loops, or excessive database calls**?

        ### 📏 **5. Adherence to Best Practices**
        - Does the code follow **industry-standard coding conventions** (PEP8 for Python, Airbnb for JavaScript, etc.)?
        - Is there proper **error handling & exception management**?
        - Are dependencies and third-party libraries **used efficiently**?

        ### ⚙ **6. Feature Implementation & Completeness**
        - Does the code **fully implement all required features** as per the documentation?
        - Are **all functionalities covered**, or are there any missing elements?

        ---

        ## **📌 Expected Output**
        Provide a **structured review** with the following details:
        1.  **List of identified issues** categorized by severity (Critical, Major, Minor).
        2. **Suggested improvements** for each issue.
        3. **Approval Status**:
        - ✅ **Approved**: If the code is production-ready.
        - 📝 **Needs Revisions**: If improvements are necessary before approval.

        🚀 **Your insights will help ensure high-quality, secure, and maintainable software.**
        """

        code_review_response = llm.invoke([code_review_prompt] + state["messages"])
        st.session_state.code_feedback = code_review_response.content
        print(code_review_response.content)
        return {"messages":code_review_response.content,"code_quality_score":code_review_response.content}
    

    def fix_code_after_security(state: State):
        st.session_state.current_step = "Fix Code After Security Review"
        """Fixes the code based on security review feedback."""
        fix_security_prompt = f"""
        You are an expert security engineer conducting a security assessment of the provided code.  

        ### **Project Context:**  
        - **Project Name:** {state["project_name"]}  
        - **Project Description:** {state["project_description"]}  

        ### **Code Under Review:** 
        {state["generated_code"]}

        ### **Identified Security Issues:**  
        {state["security_feedback"]}  

    ### **Your Task:**  
    Analyze the provided code and offer **general security improvement suggestions**, focusing on:  

    1. **Vulnerability Assessment:** Confirm whether all identified issues are valid and if any additional risks exist.  
    2. **Secure Coding Best Practices:** Suggest industry-standard security improvements (e.g., input validation, encryption, least privilege, secure dependencies).  
    3. **Code Maintainability & Performance:** Ensure security fixes do not introduce unnecessary complexity or performance overhead.  
    4. **Common Attack Vectors:** Highlight potential risks such as **SQL injection, XSS, CSRF, authentication flaws, and insecure dependencies**.  
    5. **General Security Guidelines:** Provide recommendations for improving the overall security posture of the project.  

    ### **Expected Output:**  
    Instead of fixing the code, provide **actionable insights** on improving security. Your response should include:  
    - **Key areas of concern** in the code.  
    - **Best practices** for addressing vulnerabilities.  
    - **General security principles** applicable to the project.  

    Ensure that your recommendations are clear, concise, and follow **secure coding principles**.
    """
        fix_security_response = llm.invoke([fix_security_prompt] + state["messages"])
        st.session_state.security_review_response = fix_security_response.content
        return {"messages":fix_security_response.content,"security_review_response":fix_security_response.content}


    def write_test_cases(state: State):
        st.session_state.current_step = "Write Test Cases"
        """Generates comprehensive test cases for the code."""
        test_cases_prompt = f"""
        You are an expert QA engineer specializing in test case development. Your goal is to generate **comprehensive, high-quality test cases** that ensure full code coverage and reliability.

        ### **Project Context:**  
        - **Project Name:** {state["project_name"]}  
        - **Project Description:** {state["project_description"]}  

        ### **Code to Test:** 
        {state["generated_code"]}

        ### **Existing Test Feedback (if available):**  
        {state.get("test_cases_response", "")}  

        ### **Test Case Requirements:**  
        1. **Unit Tests:** Cover all individual functions/methods with **clear assertions**.  
        2. **Integration Tests:** Validate interactions between components and dependencies.  
        3. **Edge Cases & Boundary Testing:** Include tests for extreme input values and uncommon scenarios.  
        4. **Error Handling & Exception Tests:** Ensure failures and incorrect inputs are managed properly.  
        5. **Positive & Negative Scenarios:** Cover both expected and unexpected behaviors.  
        6. **Performance Considerations:** If applicable, suggest test cases to assess efficiency and scalability.  
        7. **Clarity & Documentation:** Write **clear test descriptions**, expected results, and organize tests logically.  

        ### **Additional Considerations:**  
        - If feedback exists, incorporate necessary improvements in new test cases.  
        - Ensure best practices are followed for the relevant testing framework and language.  
        - Suggest improvements to **test structure, maintainability, and execution efficiency**.  

        Please provide **structured test cases** in the most appropriate testing framework based on the code language.
"""

        write_test_cases_response = llm.invoke([test_cases_prompt] + state["messages"])
        st.session_state.write_test_cases_response = write_test_cases_response.content
        return {"messages": write_test_cases_response.content, "write_test_cases_response": write_test_cases_response.content}


    def fix_test_cases_after_review(state: State):
        st.session_state.current_step = "Fix Test Cases After Review"
        """Fixes test cases based on review feedback."""
        fix_test_cases_prompt = f"""
        You are an expert QA engineer conducting a review of test cases to improve their effectiveness and alignment with best practices.

        ### **Project Context:**  
        - **Project Name:** {state["project_name"]}  
        - **Project Description:** {state["project_description"]}  

        ### **Original Test Cases:**  
        {state["write_test_cases_response"]}

        ### **Review Feedback:**  
        {state["test_cases_feedback"]}  

        ### **Your Task:**  
        Analyze the provided test cases and offer **constructive suggestions for improvement**, focusing on:  

        1. **Coverage Gaps:** Identify missing test scenarios, including functional, integration, and regression cases.  
        2. **Edge Cases & Error Handling:** Recommend additional test cases for boundary conditions, invalid inputs, and failure scenarios.  
        3. **Clarity & Documentation:** Suggest improvements to test descriptions, assertions, and expected outcomes for better readability.  
        4. **Test Optimization:** Identify redundant or inefficient test cases and propose refinements to improve execution speed and maintainability.  
        5. **Framework Best Practices:** Provide recommendations to align with industry best practices for the chosen testing framework.  

        ### **Expected Output:**  
        Instead of rewriting test cases, provide **detailed recommendations** on:  
        - **Key areas of improvement** in existing test cases.  
        - **Missing tests or scenarios** that should be included.  
        - **Refinements to improve clarity, efficiency, and maintainability**.  
        - **Best practices for structuring high-quality test cases**.  

        Ensure that your suggestions are **clear, actionable, and aligned with software testing standards**.
        """

        fix_test_cases_response = llm.invoke([fix_test_cases_prompt] + state["messages"])
        st.session_state.test_cases_response = fix_test_cases_response.content
        return {"messages": fix_test_cases_response.content, "test_cases_response": fix_test_cases_response.content}


    def fix_code_after_qa_feedback(state: State):
        st.session_state.current_step = "Fix Code After QA Feedback"
        """Fixes code based on QA testing feedback."""
        qa_fix_prompt = f"""
        You are an expert software engineer tasked with reviewing QA feedback and providing improvement suggestions.

        **Project Name**: {state["project_name"]}
        **Project Description**: {state["project_description"]}

        **QA Testing Feedback**:
        {state["qa_testing_feedback"]}

        **Your Task**:
        - Analyze the QA feedback and identify key issues
        - Provide specific recommendations for fixing functional defects
        - Suggest improvements for error handling and exception management
        - Highlight potential optimizations for performance-related concerns
        - Ensure recommendations align with best coding practices
        - Identify any gaps in existing test coverage and suggest additional tests

        Your output should contain **detailed suggestions** for each issue reported, ensuring the development team can make precise improvements without ambiguity.
"""


        fix_qa_response = llm.invoke([qa_fix_prompt] + state["messages"])
        st.session_state.qa_final_feedback = fix_qa_response.content
        return {"messages": fix_qa_response.content, "qa_final_feedback": fix_qa_response.content}
                
    
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
    builder.add_node("Code Review",code_review)
    builder.add_node("Fix Code After Code Review", fix_code_after_code_review)
    builder.add_node("Security Review",security_review)
    builder.add_node("Fix Code After Security Review", fix_code_after_security)
    builder.add_node("Write Test Cases", write_test_cases)
    builder.add_node("Test Cases Review", test_cases_review)
    builder.add_node("Fix Test Cases After Review", fix_test_cases_after_review)
    builder.add_node("QA Testing", qa_testing)
    builder.add_node("Fix Code After QA", fix_code_after_qa_feedback)

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

    builder.add_edge("Generate Code", "Code Review")

    builder.add_conditional_edges(
        "Code Review",
        route_code_review_decision,
        {
            "Approved": "Security Review",
            "Feedback": "Fix Code After Code Review"
        }
    )

    builder.add_edge("Fix Code After Code Review", "Generate Code")


    builder.add_conditional_edges(
        "Security Review",
        route_security_review_decision,
        {
            "Approved": "Write Test Cases",
            "Feedback": "Fix Code After Security Review"
        }
    )

    builder.add_edge("Fix Code After Security Review", "Security Review")
    builder.add_edge("Write Test Cases", "Test Cases Review")

    builder.add_conditional_edges(
        "Test Cases Review",
        route_test_cases_decision,
        {
            "Approved": "QA Testing",
            "Feedback": "Fix Test Cases After Review"
        }
    )

    builder.add_edge("Fix Test Cases After Review", "Write Test Cases")



    # Then add the conditional edges for QA testing
    builder.add_conditional_edges(
        "QA Testing",
        route_qa_testing_decision,
        {
            "Passed": END,  # If QA passes, end the workflow
            "Failed": "Fix Code After QA"  # If QA fails, go to fix code
        }
    )


    builder.add_edge("Fix Code After QA", "Generate Code")

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
        "Generate Code",
        "Code Review",
        "Fix Code After Code Review",
        "Security Review",
        "Fix Code After Security Review",
        "Write Test Cases",
        "Test Cases Review",
        "Fix Test Cases After Review",
        "QA Testing",
        "Fix Code After QA Feedback"
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
                for event in workflow.stream(inputs,config={"recursion_limit": 50}):
                    # Debug output if needed
                    # st.write(event)
                    pass
                
                st.success("Workflow completed successfully!")
                st.session_state.workflow_complete = True
                
        # Display results in tabs
        if st.session_state.user_stories:
            tabs = st.tabs(["User Stories", "Documentation", "Generated Code","Fix Code After Code Review","Fix Code After Security","Fix Test Cases After Review","Fix Code After QA Feedback"])
            
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
                        print(st.session_state.technical_documentation)
                        st.markdown(st.session_state.technical_documentation)
                        create_download_link(
                            st.session_state.technical_documentation,
                            f"{st.session_state.project_name.lower().replace(' ', '_')}_technical_documentation.md",
                            "Download Technical Documentation"
                        )
                
                if st.session_state.functional_documentation:
                    with st.expander("📋 Functional Documentation", expanded=True):
                        print(st.session_state.functional_documentation)
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

            # Fix Code After Code Review Tab
            with tabs[3]:
                st.header("Fix Code After Code Review")

                if st.session_state.code_feedback:
                    with st.expander("Code Review Feedback", expanded=True):
                        print("\n")
                        print(st.session_state.code_feedback)
                        st.info(st.session_state.code_feedback)
                
                if st.session_state.generated_code:
                    with st.expander("Fixed Code After Review", expanded=True):
                        st.code(st.session_state.generated_code)
                        create_download_link(
                            st.session_state.fixed_code_after_code_review,
                            "fixed_code_after_review.py",
                            "Download Fixed Code"
                        )

            # Fix Code After Security Review Tab
            with tabs[4]:
                st.header("Fix Code After Security Review")
                
                if st.session_state.security_review_response:
                    with st.expander("Security Review Feedback", expanded=True):
                        print("\n")
                        print(st.session_state.security_review_response)
                        st.info(st.session_state.security_review_response)
                
                if st.session_state.generated_code:
                    with st.expander("Fixed Code After Security Review", expanded=True):
                        print("\n")
                        print(st.session_state.generated_code)
                        st.code(st.session_state.generated_code)
                        create_download_link(
                            st.session_state.generated_code,
                            "fixed_code_after_security.py",
                            "Download Fixed Code"
                        )

            # Fix Test Cases After Review Tab
            with tabs[5]:
                st.header("Fix Test Cases After Review")
                
                if st.session_state.write_test_cases_response:
                    with st.expander("Test Cases to Review", expanded=True):
                        st.markdown(st.session_state.write_test_cases_response)
                
                
                if st.session_state.test_cases_response:
                    with st.expander("Fixed Test Cases After Review", expanded=True):
                        st.markdown(st.session_state.test_cases_response)
                        create_download_link(
                            st.session_state.test_cases_response,
                            "fixed_test_cases_after_review.md",
                            "Download Fixed Test Cases"
                        )
                
                if st.session_state.test_cases_feedback:
                    with st.expander("Test Cases Review Feedback", expanded=True):
                        st.info(st.session_state.test_cases_feedback)
        
            # Fix Code After QA Feedback Tab
            with tabs[6]:
                st.header("Fix Code After QA Feedback")
                
                if st.session_state.qa_testing_feedback:
                    with st.expander("QA Testing Feedback", expanded=True):
                        print(st.session_state.qa_testing_feedback)
                        st.info(st.session_state.qa_final_feedback)
                
                if st.session_state.generated_code:
                    with st.expander("Fixed Code After QA Feedback", expanded=True):
                        st.code(st.session_state.generated_code)
                        create_download_link(
                            st.session_state.generated_code,
                            "fixed_code_after_qa.py",
                            "Download Fixed Code"
                        )

            



if __name__ == "__main__":
    main()