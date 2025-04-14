import cv2
import numpy as np
from utils.image_utils import cv2_to_streamlit_image, depth_to_heatmap
import streamlit as st

def display_step_images(step_container, augmented_view, depth_map, new_view=None, width=300):
    """Display images for a single step"""
    col1, col2, col3 = step_container.columns(3)
    
    with col1:
        if augmented_view is not None and augmented_view.size > 0:
            step_container.image(
                cv2_to_streamlit_image(augmented_view), 
                caption="Augmented View with Action Paths", 
                width=width
            )
        else:
            raise ValueError("No augmented view available")
    
    with col2:
        if depth_map is not None and depth_map.size > 0:
            depth_heatmap = depth_to_heatmap(depth_map)
            step_container.image(
                cv2_to_streamlit_image(depth_heatmap), 
                caption="Depth Map", 
                width=width
            )
        else:
            raise ValueError("No depth map available")
    
    with col3:
        if new_view is not None and new_view.size > 0:
            step_container.image(
                cv2_to_streamlit_image(new_view), 
                caption="After Action View", 
                width=width
            )
        else:
            raise ValueError("No new view available")

def display_completion_check(step_container, check):
    """Display completion check information"""
    status = "✅ Completed" if check['completed'] else "❌ Not Completed"
    step_container.markdown(f"Task Completion Check: {status}")
    step_container.markdown(f"Reasoning: {check['reasoning']}")
    
    if 'images' in check and check['images']:
        step_container.markdown("**Completion Check Images:**")
        cols = step_container.columns(len(check['images']))
        for col, (img_idx, img) in zip(cols, enumerate(check['images'])):
            with col:
                st.image(
                    cv2_to_streamlit_image(img), 
                    caption=f"View {img_idx + 1}", 
                    width=200
                )
        
        if 'prompt' in check:
            step_container.markdown("**Completion Check Prompt:**")
            step_container.code(check['prompt'], language=None)
        if 'response' in check:
            step_container.markdown("**Completion Check Response:**")
            step_container.code(check['response'], language=None)

def display_failed_action(container, failed_action):
    """Display failed action information"""
    container.markdown(f"### Failed Action at Step {failed_action['step']} - {failed_action['type']}")
    container.error(f"Error: {failed_action['error']}")
    
    if 'step' in failed_action and failed_action['step'] < len(failed_action.get('augmented_images', [])):
        container.image(
            cv2_to_streamlit_image(failed_action['augmented_images'][failed_action['step']]), 
            caption=f"Prompt Image for Failed Action at Step {failed_action['step']}", 
            width=300
        )
    
    if 'vlm_prompt' in failed_action:
        container.markdown("**VLM Prompt:**")
        container.code(failed_action['vlm_prompt'], language=None)
    if 'vlm_response' in failed_action:
        container.markdown("**VLM Response:**")
        container.code(failed_action['vlm_response'], language=None)
    if 'action_number' in failed_action:
        container.markdown(f"**Action Number:** {failed_action['action_number']}")
    container.markdown("---")

def draw_action_overlay(image, action_info):
    """Draw action overlay on image"""
    if action_info and "center_position" in action_info:
        center_x, center_y = action_info["center_position"]
        # Draw a larger green circle with outline (thickness=2) instead of fill
        cv2.circle(image, (center_x, center_y), 20, (0, 255, 0), 2)
        
        # If there's a boundary point, draw it in red
        if action_info.get("boundary_point") is not None:
            boundary_x, boundary_y = action_info["boundary_point"]
            cv2.circle(image, (boundary_x, boundary_y), 10, (0, 0, 255), 2)
    return image 