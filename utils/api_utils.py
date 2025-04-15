import requests
import json
import re
import json_repair
from utils.image_utils import convert_image_to_base64
import cv2
import numpy as np
import base64

def send_api_request(image_base64, api_url, min_angle=20, number_size=30, min_path_length=80, min_arrow_width=60):
    """Send a request to the action proposal API."""
    payload = {
        "image": image_base64,
        "min_angle": min_angle,
        "number_size": number_size,
        "min_path_length": min_path_length,
        "min_arrow_width": min_arrow_width
    }
    response = requests.post(api_url, json=payload)
    return response

def handle_api_response(response):
    """Handle the API response and return the augmented image and actions."""
    if response.status_code == 200:
        try:
            data = response.json()
            if "image" not in data or not data["image"]:
                raise Exception("API response missing or empty image data")
                
            augmented_image_base64 = data["image"]
            augmented_image_bytes = base64.b64decode(augmented_image_base64)
            augmented_image_np = cv2.imdecode(np.frombuffer(augmented_image_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            if augmented_image_np is None or augmented_image_np.size == 0:
                raise Exception("Failed to decode image from API response")
                
            if "actions" not in data:
                raise Exception("API response missing actions data")
                
            actions = data["actions"]

            # process navigability map
            navigability_mask = data['navigability_mask']
            navigability_mask_np = np.array(navigability_mask)

            return augmented_image_np, actions, navigability_mask_np
        except Exception as e:
            raise Exception(f"Error processing API response: {str(e)}")
    else:
        raise Exception(f"Error: API returned status code {response.status_code}")