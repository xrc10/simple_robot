import streamlit as st
import os
import json
import glob
import re
import pandas as pd
import numpy as np
from PIL import Image
import base64

st.set_page_config(layout="wide", page_title="Robot Navigation Visualizer")

def get_step_folders():
    """Get all step folders in the views directory sorted by step number."""
    step_folders = glob.glob("views/step_*")
    
    # Sort by step number
    step_folders.sort(key=lambda x: int(re.search(r'step_(\d+)', x).group(1)))
    return step_folders

def get_simulation_report():
    """Read the simulation report if available."""
    report_path = "views/simulation_report.txt"
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            return f.read()
    return None

def read_json_file(file_path):
    """Read and parse a JSON file."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return None

def read_text_file(file_path):
    """Read a text file."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return f.read()
    return "File not available"

def main():
    st.title("Robot Navigation Visualization")
    
    # Initialize session state for step tracking
    if 'step_index' not in st.session_state:
        st.session_state.step_index = 0
    
    # Check if views directory exists
    if not os.path.exists("views"):
        st.error("Views directory not found. Please run the simulation first.")
        return
    
    # Display simulation report if available
    report = get_simulation_report()
    if report:
        with st.expander("Simulation Report", expanded=True):
            st.text(report)
    
    # # Display simulation video if available
    # video_path = "views/simulation_video.mp4"
    # if os.path.exists(video_path):
    #     with st.expander("Simulation Video", expanded=True):
    #         try:
    #             # Get absolute path and verify file
    #             abs_path = os.path.abspath(video_path)
    #             if not os.path.isfile(abs_path):
    #                 st.error(f"Video file is not a regular file: {abs_path}")
    #             else:
    #                 # Try to open the file to check if it's readable
    #                 with open(abs_path, 'rb') as f:
    #                     f.read(1)  # Just check if we can read the file
                    
    #                 # Use experimental video component
    #                 video_file = open(abs_path, 'rb')
    #                 video_bytes = video_file.read()
                    
    #                 # Add HTML5 video tag with controls
    #                 st.markdown(f"""
    #                 <video width="100%" controls>
    #                     <source src="data:video/mp4;base64,{base64.b64encode(video_bytes).decode()}" type="video/mp4">
    #                     Your browser does not support the video tag.
    #                 </video>
    #                 """, unsafe_allow_html=True)
    #         except Exception as e:
    #             st.error(f"Error playing video: {str(e)}")
    #             st.info("Supported video formats: MP4, WebM, Ogg")
    #             st.info(f"Video path: {abs_path}")
    #             st.info("If the video is large, try reducing its size or converting to a different format.")
    #             st.info("Current video size: " + str(os.path.getsize(abs_path) / (1024 * 1024)) + " MB")
    # else:
    #     st.warning(f"Simulation video not found. Expected path: {os.path.abspath(video_path)}")
    
    # Get step folders
    step_folders = get_step_folders()
    
    if not step_folders:
        st.warning("No simulation steps found in the views directory.")
        return
    
    # Create a step selector
    col1, col2 = st.columns([1, 3])
    with col1:
        step_options = [f"Step {i}" for i in range(len(step_folders))]
        selected_step = st.selectbox("Select Step", step_options, index=st.session_state.step_index, key="step_selector")
        step_index = int(selected_step.split()[1])
        selected_folder = step_folders[step_index]
        
        # Add navigation buttons
        nav_col1, nav_col2 = st.columns(2)
        with nav_col1:
            if step_index > 0:
                if st.button("← Previous"):
                    st.session_state.step_index = step_index - 1
                    st.rerun()
        with nav_col2:
            if step_index < len(step_folders) - 1:
                if st.button("Next →"):
                    st.session_state.step_index = step_index + 1
                    st.rerun()
    
    # Display step visualization
    with st.container():
        # Display images in one row
        img_col1, img_col2, img_col3 = st.columns(3)
        
        # Augmented view
        aug_view_path = os.path.join(selected_folder, "augmented_view.jpg")
        if os.path.exists(aug_view_path):
            with img_col1:
                st.image(aug_view_path, caption="Augmented View", use_container_width=True)
        
        # Depth map
        depth_path = os.path.join(selected_folder, "depth.jpg")
        if os.path.exists(depth_path):
            with img_col2:
                st.image(depth_path, caption="Depth Map", use_container_width=True)
        
        # New view
        new_view_path = os.path.join(selected_folder, "new_view.jpg")
        if os.path.exists(new_view_path):
            with img_col3:
                st.image(new_view_path, caption="New View After Action", use_container_width=True)
        
        # Display combined view if available
        combined_path = os.path.join(selected_folder, "combined.jpg")
        if os.path.exists(combined_path):
            st.image(combined_path, caption="Combined View", use_container_width=True)
    
    # Display VLM information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("VLM Interaction")
        vlm_prompt = read_text_file(os.path.join(selected_folder, "vlm_prompt.txt"))
        st.text_area("VLM Prompt", vlm_prompt, height=200)
        
        vlm_output = read_text_file(os.path.join(selected_folder, "vlm_output.txt"))
        st.text_area("VLM Response", vlm_output, height=300)
    
    with col2:
        st.subheader("Navigation Results")
        parsed_result = read_json_file(os.path.join(selected_folder, "parsed_result.json"))
        
        if parsed_result:
            st.markdown(f"**Action:** {parsed_result.get('action', 'N/A')}")
            st.markdown(f"**Progress:** {parsed_result.get('progress', 'N/A')}")
            st.markdown(f"**Landmarks:** {parsed_result.get('landmarks', 'N/A')}")
            
            # Display reasoning in expandable section
            with st.expander("Reasoning"):
                st.write(parsed_result.get('reasoning', 'No reasoning available'))
        
        # Display completion information if available
        completion_parsed = read_json_file(os.path.join(selected_folder, "completion_parsed_result.json"))
        if completion_parsed:
            with st.expander("Completion Details"):
                st.json(completion_parsed)

if __name__ == "__main__":
    main()
