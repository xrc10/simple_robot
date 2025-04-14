import streamlit as st
import cv2
import os
import shutil
import base64
import requests
import json
import numpy as np
import time
import re # Import the regex module
from models import VLM
from env import ThorEnvDogView
from agent import VLMNavigationAgent
from PIL import Image, ImageDraw, ImageFont
import tempfile
from functools import lru_cache
from utils.config import setup_sidebar_config, initialize_session_state
from utils.visualization import display_step_images, display_completion_check, display_failed_action
from utils.image_utils import depth_to_heatmap
from utils.logger import logger

def create_simulation_video(agent, output_path='simulation_video.mp4'):
    """Create a video from the simulation history."""
    if not agent or not agent.augmented_images:
        st.error("No simulation history available to create video")
        return None
    
    # Get video dimensions from the first image
    height, width = agent.augmented_images[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, 1.0, (width * 2, height))  # Double width for side-by-side
    
    try:
        for i, (augmented_img, depth_img) in enumerate(zip(agent.augmented_images, agent.depth_memory)):
            # Convert depth to heatmap
            depth_heatmap = depth_to_heatmap(depth_img)
            
            # Convert BGR to RGB for PIL
            augmented_rgb = cv2.cvtColor(augmented_img, cv2.COLOR_BGR2RGB)
            depth_rgb = cv2.cvtColor(depth_heatmap, cv2.COLOR_BGR2RGB)
            
            # Create PIL images
            augmented_pil = Image.fromarray(augmented_rgb)
            depth_pil = Image.fromarray(depth_rgb)
            
            # Create a new image with double width
            combined = Image.new('RGB', (width * 2, height))
            combined.paste(augmented_pil, (0, 0))
            combined.paste(depth_pil, (width, 0))
            
            # Add text overlay
            draw = ImageDraw.Draw(combined)
            try:
                font = ImageFont.truetype("arial.ttf", 30)
            except:
                font = ImageFont.load_default()
            
            # Add step number and action
            step_text = f"Step {i+1}"
            if i < len(agent.action_memory):
                action = agent.action_memory[i]
                action_text = f"Action: {action.action_number if hasattr(action, 'action_number') else action}"
                draw.text((10, 10), step_text, (255, 255, 255), font=font)
                draw.text((10, 50), action_text, (255, 255, 255), font=font)
            
            # Convert back to OpenCV format
            combined_cv = cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)
            video.write(combined_cv)
        
        video.release()
        return output_path
    except Exception as e:
        st.error(f"Error creating video: {str(e)}")
        if video:
            video.release()
        return None

def main():
    st.set_page_config(page_title="Robot Navigation Simulation", layout="wide")
    st.title("VLM Navigation Simulation")

    # Setup configuration from sidebar
    config = setup_sidebar_config()

    # Control buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_button = st.button("Start Simulation")
    with col2:
        stop_button = st.button("Stop Simulation")

    # Initialize session state
    initialize_session_state()

    # Main display area
    main_container = st.container()

    # Start simulation
    if start_button:
        with st.spinner("Initializing simulation..."):
            # Create the environment first
            env = ThorEnvDogView(config.floor_id)
            # Create the agent with the environment
            st.session_state.agent = VLMNavigationAgent(
                env=env,
                model_id=config.model_id,
                api_url=config.api_url,
                max_distance_to_move=config.max_distance_to_move
            )
            st.session_state.running = True
            main_container.header("Simulation Started")
            
            # Display initial view and depth
            display_step_images(
                main_container,
                st.session_state.agent.view,
                st.session_state.agent.depth
            )

    # Stop simulation
    if stop_button:
        st.session_state.running = False
        main_container.warning("Simulation stopped by user")

    # Run simulation steps
    if st.session_state.running and st.session_state.agent is not None:
        # Continue until max steps or stopped
        if st.session_state.agent.step_number < config.max_steps:
            # Create a new container for each step
            step_container = main_container.container()
            step_container.subheader(f"Step {st.session_state.agent.step_number + 1}")

            try:
                # Execute one step of the navigation - updated to match main.py style
                step_result = st.session_state.agent.step(
                    config.target,
                    config.task_prompt,
                    config.max_steps
                )
                
                # Extract data from the step result
                augmented_view = step_result.get("augmented_view")
                new_view = step_result.get("new_view")
                reasoning = step_result.get("reasoning", "")
                action_chosen = step_result.get("action_chosen", "")
                is_completed = step_result.get("completed", False)
                vlm_output_str = step_result.get("vlm_output_str", "")
                progress = step_result.get("progress", "")
                landmarks = step_result.get("landmarks", "")
                failed_action = step_result.get("failed_action")

                if augmented_view is None:  # Simulation completed or max steps reached
                    st.session_state.running = False
                    if is_completed:
                        main_container.success("Task completed successfully!")
                    return

                # Display step information
                col1, col2 = step_container.columns(2)
                
                # Display images
                display_step_images(
                    step_container,
                    augmented_view,
                    st.session_state.agent.depth,
                    new_view
                )

                # Display model's raw output and choice of action
                with col2:
                    step_container.text("VLM Raw Output:")
                    step_container.code(vlm_output_str, language=None)
                    step_container.markdown("**VLM Reasoning:**")
                    step_container.info(reasoning)
                    step_container.text(f"VLM Chose Action: {action_chosen}")
                    
                    if progress:
                        step_container.markdown("**Progress:**")
                        step_container.info(progress)
                    
                    if landmarks:
                        step_container.markdown("**Landmarks:**")
                        step_container.info(landmarks)

                    # Display completion check information if available
                    if st.session_state.agent.memory.completion_checks:
                        latest_check = st.session_state.agent.memory.get_last_completion_check()
                        if latest_check:
                            display_completion_check(step_container, latest_check)
                
                # Display failed action if any
                if failed_action:
                    display_failed_action(step_container, failed_action)

                # Calculate time since last step and sleep if needed
                current_time = time.time()
                elapsed = current_time - st.session_state.last_step_time
                if elapsed < config.step_delay:
                    time.sleep(config.step_delay - elapsed)
                st.session_state.last_step_time = time.time()
                
                # Only rerun if we're not at max steps
                if st.session_state.agent.step_number < config.max_steps:
                    st.rerun()

            except Exception as e:
                step_container.error(f"Error during simulation step: {str(e)}")
                st.session_state.running = False
                return

        else:
            st.session_state.running = False
            st.session_state.agent.completed = True
            main_container.warning("Maximum steps reached.")
    
    # Display action history with images
    if st.session_state.agent is not None and st.session_state.agent.step_number > 0:
        steps_per_page = 20
        total_steps = len(st.session_state.agent.action_memory)
        total_pages = (total_steps + steps_per_page - 1) // steps_per_page

        page_number = st.sidebar.number_input("Page", min_value=1, max_value=total_pages, value=1)
        start_index = (page_number - 1) * steps_per_page
        end_index = start_index + steps_per_page

        # Add video creation button
        if st.sidebar.button("Create Simulation Video"):
            with st.spinner("Creating video..."):
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                    video_path = create_simulation_video(st.session_state.agent, tmp_file.name)
                    if video_path:
                        with open(video_path, 'rb') as f:
                            video_bytes = f.read()
                            st.sidebar.download_button(
                                label="Download Simulation Video",
                                data=video_bytes,
                                file_name="simulation_video.mp4",
                                mime="video/mp4"
                            )

        # Display failed actions
        if st.session_state.agent.failed_actions:
            st.subheader("Failed Actions")
            for failed_action in st.session_state.agent.failed_actions:
                display_failed_action(st, failed_action)

        # Display successful actions and completions
        st.subheader("Action History")
        for i, action_record in enumerate(st.session_state.agent.action_memory[start_index:end_index], start=start_index):
            # Create a container for each step
            step_container = st.container()
            
            # Display the action information
            step_container.markdown(f"### Step {i}: {action_record.action if hasattr(action_record, 'action') else action_record}")
            
            # Display images for this step if available
            if i < len(st.session_state.agent.augmented_images):
                display_step_images(
                    step_container,
                    st.session_state.agent.augmented_images[i],
                    st.session_state.agent.depth_memory[i],
                    st.session_state.agent.complete_images[i]
                )
            
            step_container.markdown("---")  # Add a separator between steps

if __name__ == "__main__":
    main() 