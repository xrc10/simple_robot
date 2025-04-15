import requests
from urllib.parse import urljoin
import numpy as np
import cv2
from PIL import Image
import base64


class UnitreeDogEnv(object):
    def __init__(self, request_url, camera_fov=80) -> None:
        self.request_url = request_url
        self.last_event = None
        self.camera_fov = camera_fov
        # Initial position setup - similar to crouch + lookdown in ThorEnvDogView
        self._init_position()
    
    def _init_position(self):
        """Initialize position, similar to the crouch and lookdown in ThorEnvDogView"""
        # In the real dog, we might not need this or could use a specific movement
        pass
    
    def reset(self, request_url=None):
        """Reset the environment"""
        if request_url:
            self.request_url = request_url
        self._init_position()
        
    def get_last_event(self):
        """Return the last event"""
        return self.last_event
    
    def step(self, action, degrees=30, magnitude=0.25):
        """
        Take a step in the environment
        - action: string describing the action
        - degrees: for rotation actions
        - magnitude: for movement actions
        """
        result = {}
        
        if 'Rotate' in action:
            # Convert degrees to radians for vyaw
            # Positive degrees = counterclockwise, so we negate for vyaw
            vyaw = -degrees * (3.14159 / 180)
            result = self._move(0, 0, vyaw)
        elif 'Move' in action:
            if 'Ahead' in action:
                result = self._move(magnitude, 0, 0)
            elif 'Back' in action:
                result = self._move(-magnitude, 0, 0)
            elif 'Left' in action:
                result = self._move(0, magnitude, 0)
            elif 'Right' in action:
                result = self._move(0, -magnitude, 0)
        
        # Update last_event
        self.last_event = result
        return result
    
    def _move(self, vx=0, vy=0, vyaw=0):
        """Send movement command to the robot"""
        url = urljoin(self.request_url, "/signalservice/robot/move")
        try:
            response = requests.post(url, json={"vx": vx, "vy": vy, "vyaw": vyaw}).json()
            if response["code"] != '0':
                raise Exception(f"Robot move failed: {response['message']}")
            return response
        except Exception as e:
            return {"error": str(e)}
    
    def _request_snapshot(self):
        """Request a snapshot from the robot's camera"""
        url = urljoin(self.request_url, "/signalservice/video/color_depth_snapshot")
        try:
            response = requests.get(url).json()
            if response["code"] != '0':
                raise Exception(f"Robot snapshot failed [{response['code']}]: {response['message']}")
            
            result = {}
            
            # Get RGB image
            rgb_width = response['data'].get("rgb_width")
            rgb_height = response['data'].get("rgb_height")
            rgb_base64 = response['data'].get("rgb_data")
            
            if rgb_base64 and rgb_width and rgb_height:
                rgb_bytes = base64.b64decode(rgb_base64)
                rgb_image = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape((rgb_height, rgb_width, 3))
                result["frame"] = rgb_image
            
            # Get depth image
            depth_width = response['data'].get("depth_width")
            depth_height = response['data'].get("depth_height")
            depth_base64 = response['data'].get("depth_data")
            
            if depth_base64 and depth_width and depth_height:
                depth_bytes = base64.b64decode(depth_base64)
                depth_image = np.frombuffer(depth_bytes, dtype=np.uint16).reshape((depth_height, depth_width))
                result["depth_frame"] = depth_image
                
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def get_observation(self):
        """Get the current RGB observation"""
        result = self._request_snapshot()
        if "frame" in result:
            return result["frame"]
        return None
    
    def get_depth(self):
        """Get the current depth observation"""
        result = self._request_snapshot()
        if "depth_frame" in result:
            return result["depth_frame"]
        return None
