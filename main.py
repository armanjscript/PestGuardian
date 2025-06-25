import os
import asyncio
import streamlit as st
from ultralytics import YOLO
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_ollama import OllamaLLM
from langgraph.graph import Graph
from typing import Dict, TypedDict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
import nest_asyncio

# Apply nest_asyncio early to handle async issues
nest_asyncio.apply()

# Set environment variables
os.environ["SERPER_API_KEY"] = "YOUR_SERPER_API_KEY"

def save_uploaded_file(uploaded_file):
    """Save uploaded file in the current directory with its original name and extension."""
    try:
        file_name = uploaded_file.name
        file_path = os.path.join(os.getcwd(), file_name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        print(f"Saved image to: {file_path}")
        return file_name
    except Exception as e:
        st.error(f"Error saving image: {str(e)}")
        return None

# Initialize models and tools
def initialize_components():
    """Initialize all necessary components"""
    try:
        # YOLO Pest Detection Model
        model_path = os.path.join('best.pt')
        yolo_model = YOLO(model_path)
        
        # Ollama LLM with Qwen2.5
        llm = OllamaLLM(model="qwen2.5")
        
        # Serper API for web search
        serper = GoogleSerperAPIWrapper()
        
        return yolo_model, llm, serper
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        return None, None, None

# Define the agent state with Optional fields
class AgentState(TypedDict):
    """State of our pest detection agent"""
    image_path: str
    detected_pests: Optional[List[str]]
    search_results: Optional[Dict[str, str]]
    recommendations: Optional[str]

# Define nodes for the LangGraph workflow
def detect_pests(state: AgentState) -> AgentState:
    """Detect pests in the uploaded image using YOLO"""
    try:
        if "image_path" not in state or not state["image_path"]:
            raise ValueError("No image path provided")
            
        image_path = state["image_path"]
        
        # Run YOLO detection
        results = yolo_model.predict(image_path)
        print(results)
        
        # Extract pest labels
        detected_pests = []
        for result in results:
            # For classification models, use probs instead of boxes
            if hasattr(result, 'probs'):
                top5 = result.probs.top5
                confidences = result.probs.top5conf.tolist()
                for i, class_id in enumerate(top5):
                    pest_label = yolo_model.names[class_id]
                    if confidences[i] > 0.1:  # Only include if confidence > 10%
                        detected_pests.append(pest_label)
        
        return {**state, "detected_pests": list(set(detected_pests))}  # Remove duplicates
    except Exception as e:
        st.error(f"Error in pest detection: {str(e)}")
        return {**state, "detected_pests": []}

def search_elimination_methods(state: AgentState) -> AgentState:
    """Search for pest elimination methods using Serper API"""
    try:
        if "detected_pests" not in state or not state["detected_pests"]:
            return {**state, "search_results": {}}
            
        pests = state["detected_pests"]
        search_results = {}
        
        for pest in pests:
            query = f"best ways to eliminate {pest} in garden"
            results = serper.run(query)
            search_results[pest] = results
        
        return {**state, "search_results": search_results}
    except Exception as e:
        st.error(f"Error in search: {str(e)}")
        return {**state, "search_results": {}}

def generate_recommendations(state: AgentState) -> AgentState:
    """Generate recommendations using Qwen2.5"""
    try:
        if "detected_pests" not in state or not state["detected_pests"]:
            return {**state, "recommendations": "No pests detected to generate recommendations"}
            
        if "search_results" not in state:
            return {**state, "recommendations": "No search results available"}
            
        pests = state["detected_pests"]
        search_data = state["search_results"]
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an agricultural expert helping farmers eliminate pests. Provide concise, practical recommendations based on the following information."),
            ("human", "Detected pests: {pests}\n\nSearch results: {search_data}")
        ])
        
        chain = prompt_template | llm
        recommendations = chain.invoke({
            "pests": ", ".join(pests) if pests else "None detected",
            "search_data": str(search_data) if search_data else "No search results"
        })
        
        return {**state, "recommendations": str(recommendations)}
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return {**state, "recommendations": "Failed to generate recommendations"}

# Build the workflow graph
def build_workflow():
    """Build the LangGraph workflow"""
    workflow = Graph()
    
    # Add nodes
    workflow.add_node("detect_pests", detect_pests)
    workflow.add_node("search_methods", search_elimination_methods)
    workflow.add_node("generate_recommendations", generate_recommendations)
    
    # Define edges
    workflow.add_edge("detect_pests", "search_methods")
    workflow.add_edge("search_methods", "generate_recommendations")
    
    # Set entry and exit points
    workflow.set_entry_point("detect_pests")
    workflow.set_finish_point("generate_recommendations")
    
    return workflow.compile()

# Streamlit UI
def main():
    st.title("🌱 Pest Detection & Elimination Assistant")
    st.markdown("Upload an image of pests in your garden/farm to get detection and elimination recommendations")
    
    # Initialize components
    global yolo_model, llm, serper
    yolo_model, llm, serper = initialize_components()
    
    if yolo_model is None or llm is None or serper is None:
        st.error("Failed to initialize required components. Please check your setup.")
        return
    
    # Build workflow
    pest_workflow = build_workflow()
    
    # File upload
    uploaded_file = st.file_uploader("Upload pest image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_name = save_uploaded_file(uploaded_file)
        if not file_name:
            return
        
        # Display image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        # Run workflow
        with st.spinner("Analyzing image and generating recommendations..."):
            try:
                # Initialize state with all required fields
                initial_state = AgentState(
                    image_path=file_name,
                    detected_pests=None,
                    search_results=None,
                    recommendations=None
                )
                
                # Execute workflow
                result = pest_workflow.invoke(initial_state)
                
                # Display results
                st.subheader("🔍 Detection Results")
                if result.get("detected_pests"):
                    st.success(f"Detected pests: {', '.join(result['detected_pests'])}")
                else:
                    st.warning("No pests detected in the image")
                
                st.subheader("💡 Recommended Solutions")
                st.write(result.get("recommendations", "No recommendations available"))
                
                # Show search results (collapsible)
                if result.get("search_results"):
                    with st.expander("View detailed search results"):
                        for pest, results in result["search_results"].items():
                            st.markdown(f"**{pest}**")
                            st.write(results)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                # Clean up saved file
                if file_name and os.path.exists(file_name):
                    try:
                        os.remove(file_name)
                    except Exception as e:
                        st.warning(f"Could not delete temporary file: {str(e)}")

if __name__ == "__main__":
    # Create a new event loop for Streamlit
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        main()
    finally:
        loop.close()