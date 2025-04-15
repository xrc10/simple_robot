import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils.image_utils import depth_to_heatmap, cv2_to_streamlit_image
from utils.logger import logger

def create_simulation_video(agent, output_path='simulation_video.mp4'):
    """Create a video from the simulation history."""
    if not agent or not agent.augmented_images:
        logger.error("No simulation history available to create video")
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
                action_text = f"Action: {action.action_number if action.action_number is not None else 'None'}"
                draw.text((10, 10), step_text, (255, 255, 255), font=font)
                draw.text((10, 50), action_text, (255, 255, 255), font=font)
                
                # Add reasoning if available
                if hasattr(action, 'reasoning') and action.reasoning:
                    # Split reasoning into multiple lines if too long
                    max_chars_per_line = 60
                    reasoning_lines = [action.reasoning[i:i+max_chars_per_line] for i in range(0, len(action.reasoning), max_chars_per_line)]
                    for j, line in enumerate(reasoning_lines):
                        draw.text((10, 90 + j*40), f"Reasoning: {line}", (255, 255, 255), font=font)
            
            # Convert back to OpenCV format
            combined_cv = cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)
            video.write(combined_cv)
        
        video.release()
        logger.info(f"Successfully created simulation video: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error creating video: {str(e)}")
        if video:
            video.release()
        return None 