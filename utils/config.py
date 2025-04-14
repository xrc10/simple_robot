import streamlit as st
from dataclasses import dataclass
from typing import Optional

@dataclass
class SimulationConfig:
    """Configuration for simulation"""
    floor_id: str
    model_id: str
    api_url: str
    max_distance_to_move: float
    target: str
    task_prompt: str
    step_delay: float
    max_steps: int

def setup_sidebar_config() -> SimulationConfig:
    """Setup and return simulation configuration from sidebar inputs"""
    st.sidebar.header("Simulation Settings")
    
    # Floor plan selection
    floor_id = st.sidebar.text_input(
        "Select Floor Plan", 
        value="FloorPlan_Train1_5"
    )
    
    # Model selection
    model_id = st.sidebar.selectbox(
        "Model ID", 
        options=[
            "Pro/Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/Qwen2.5-VL-32B-Instruct",
            "Qwen/Qwen2.5-VL-72B-Instruct"
        ],
        index=0
    )
    
    # API URL
    api_url = st.sidebar.text_input(
        "Action Proposal API URL", 
        value="http://10.8.25.28:8075/generate_action_proposals"
    )
    
    # Max distance to move
    max_distance_to_move = st.sidebar.number_input(
        "Maximum Distance to Move (meters)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1
    )
    
    # Target description
    target = st.sidebar.text_area(
        "Navigation Target", 
        value="find the TV"
    )
    
    # Task prompt template
    task_prompt = st.sidebar.text_area(
        "Task Prompt", 
        value="""You are a robot navigating a house. You see an image with numbered paths representing possible directions.
Your task is to follow these navigation instructions: '{TARGET}'
First, briefly describe what you see in the current view (e.g., "I see a kitchen with a counter and cabinets").
Then analyze the available paths ({ACTIONS}) and choose the best path number to follow the instructions.
If no path seems helpful, choose '0' to turn around.
Output your response as a JSON object with two keys: "reasoning" (your description and reasoning) and "action" (the chosen number as a string or '0').
Example: {{"reasoning": "I see a kitchen with a counter and cabinets. The instructions say to go left and then find the fridge. Path 3 leads to the left, which matches the first part of the instructions.", "action": "3"}}
Example: {{"reasoning": "I see a dead end with no clear paths forward. I should turn around to explore other directions.", "action": "0"}}"""
    )
    
    # Step delay
    step_delay = st.sidebar.slider(
        "Delay between steps (seconds)", 
        min_value=0.0, 
        max_value=5.0, 
        value=0.0, 
        step=0.1
    )
    
    # Maximum steps
    max_steps = st.sidebar.number_input(
        "Maximum Steps", 
        min_value=1, 
        max_value=100, 
        value=50
    )
    
    return SimulationConfig(
        floor_id=floor_id,
        model_id=model_id,
        api_url=api_url,
        max_distance_to_move=max_distance_to_move,
        target=target,
        task_prompt=task_prompt,
        step_delay=step_delay,
        max_steps=max_steps
    )

def initialize_session_state():
    """Initialize or reset the session state"""
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'vlm_output_str' not in st.session_state:
        st.session_state.vlm_output_str = "No output available."
    if 'last_step_time' not in st.session_state:
        st.session_state.last_step_time = 0 