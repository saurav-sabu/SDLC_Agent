import streamlit as st
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4o")

# Define State
class State(TypedDict):
    messages: Annotated[Optional[List], add_messages]
    project_name: str
    project_description: str
    features: List[str]
    product_decision: str
    feedback: str

# Define Product Owner Review Schema
class ProductOwnerRoute(BaseModel):
    step: Literal["Approved", "Feedback"] = Field(description="Routing Decision")
    feedback: str = Field(description="Feedback if required")

router_product_owner_route = llm.with_structured_output(ProductOwnerRoute)

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
    - Features: {state["features"]}
    
    Generate at least one user story per feature.
    """
    
    messages = state.get("messages", [])
    response = llm.invoke([user_story_prompt] + messages)
    messages.append(response)
    
    return {"messages": messages}

def product_owner_review(state: State):
    message_content = state["messages"][-1].content  # Extract last response
    decision = router_product_owner_route.invoke([
        SystemMessage(content="Review the user stories for approval or feedback."),
        HumanMessage(content=message_content),
    ])
    return {"product_decision": decision.step, "feedback": decision.feedback}

def route_product_decision(state: State):
    return "Revise User Stories" if state["product_decision"] == "Feedback" else END

def revise_user_stories(state: State):
    revise_prompt = f"""
    Improve the user stories based on the following feedback:
    {state["feedback"]}
    
    **Project Details:**
    - Name: {state["project_name"]}
    - Description: {state["project_description"]}
    - Features: {state["features"]}
    
    Original User Stories:
    {state["messages"][-1].content}
    """
    
    messages = state.get("messages", [])
    revised_response = llm.invoke([revise_prompt] + messages)
    messages.append(revised_response)
    
    return {"messages": messages}

# Create workflow graph
builder = StateGraph(State)
builder.add_node("Generate User Stories", generate_user_stories)
builder.add_node("Product Owner Review", product_owner_review)
builder.add_node("Revise User Stories", revise_user_stories)

builder.add_edge(START, "Generate User Stories")
builder.add_edge("Generate User Stories", "Product Owner Review")
builder.add_edge("Revise User Stories", "Product Owner Review")

builder.add_conditional_edges("Product Owner Review", route_product_decision, {"Revise User Stories": "Revise User Stories", END: END})

graph = builder.compile()

# Streamlit UI
st.title("Agile User Story Generator")

project_name = st.text_input("Project Name")
project_description = st.text_area("Project Description")
features = st.text_area("List Features (comma-separated)")

if st.button("Generate User Stories"):
    feature_list = [f.strip() for f in features.split(",") if f.strip()]
    test_state = {"project_name": project_name, "project_description": project_description, "features": feature_list, "messages": [], "product_decision": "", "feedback": ""}
    response = graph.invoke(test_state)
    
    st.subheader("Initial User Stories")
    st.write(response["messages"][0].content)
    
    st.subheader("Feedback")
    st.write(response["feedback"])
    
    if response["product_decision"] == "Feedback":
        st.subheader("Final User Stories (Revised)")
        st.write(response["messages"][-1].content)
    else:
        st.success("User stories approved!")
