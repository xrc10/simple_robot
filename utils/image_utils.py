import cv2
import numpy as np
import base64
from PIL import Image, ImageDraw, ImageFont

def convert_image_to_base64(image):
    """Convert an image to base64 format."""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def cv2_to_streamlit_image(image):
    """Convert OpenCV image to format suitable for Streamlit display"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def depth_to_heatmap(depth):
    """Convert depth array to heatmap visualization"""
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

def create_combined_image(augmented_img, depth_img, width, height):
    """Create a combined image with augmented view and depth map side by side"""
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
    
    return combined

def add_text_overlay(image, text, position=(10, 10), color=(255, 255, 255)):
    """Add text overlay to an image"""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, color, font=font)
    return image 