from pathlib import Path
from urllib.parse import urljoin
from typing import Any, Dict, Optional, Union, List
import traceback
import requests
import base64

import cv2
import numpy as np
from pydantic import field_validator
from PIL import Image

from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry
from omagent_core.tool_system.base import ArgSchema, BaseTool
from ..schemas.note import Note, VisionState, Step

CURRENT_PATH = Path(__file__).parents[0]

SNAPSHOT_SUFFIX = "/signalservice/video/color_depth_snapshot"

ARGSCHEMA = {
}


@registry.register_tool()
class GetImageSample(BaseTool):
    """Tool for making Unitree Go2 robot to get image sample."""

    class Config:
        """Configuration for this pydantic object."""

        extra = "allow"
        arbitrary_types_allowed = True

    args_schema: ArgSchema = ArgSchema(**ARGSCHEMA)
    description: str = "Get the image sample of the current view of the robot dog."
    request_url: str
    
    def _request_snapshot(self) -> Dict[str, Any]:
        url = urljoin(self.request_url, SNAPSHOT_SUFFIX)
        response = requests.get(url).json()
        if response["code"] != '0':
            raise Exception(f"Robot snapshot failed [{response['code']}]: {response['message']}")
        rgb_width = response['data'].get("rgb_width")
        rgb_height = response['data'].get("rgb_height")
        rgb_base64 = response['data'].get("rgb_data")

        depth_width = response['data'].get("depth_width")
        depth_height = response['data'].get("depth_height")
        depth_base64 = response['data'].get("depth_data")
        
        if depth_base64 and depth_width and depth_height:
            # 将base64图像数据转换为numpy数组
            depth_bytes = base64.b64decode(depth_base64)
            depth_image = np.frombuffer(depth_bytes, dtype=np.uint16).reshape((depth_height, depth_width))
            
            # 将深度数据归一化到0-255范围
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        if rgb_base64 and rgb_width and rgb_height:
            rgb_bytes = base64.b64decode(rgb_base64)
            rgb_image = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape((rgb_height, rgb_width, 3))
            return Image.fromarray(rgb_image)
        else:
            raise Exception("Failed to get snapshot")
    
    def take_shot(self):
        pil_image = self._request_snapshot()
        
        # Resize image to have longest edge as 512 pixels while maintaining aspect ratio
        width, height = pil_image.size
        max_dim = max(width, height)
        if max_dim > 512:
            scale_factor = 512 / max_dim
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
            
        return pil_image
    
    def update_memory(self, vision_states: List[VisionState]):
        cache_data: Note = self.stm(self.workflow_instance_id)["note"]
        cache_data.current_step().vision = vision_states
        self.stm(self.workflow_instance_id)["robot_memory"] = cache_data

    def _run(self, memorize: bool = True) -> Dict[str, Any]:
        """
        Control the Unitree Go2 robot to get image sample.
        """

        try:
            image = self.take_shot()
            vision_states = [VisionState(image=image, vyaw=0)]
            if memorize:
                self.update_memory(vision_states)

            return {
                "code": 0,
                "msg": "success",
                "result": "Successfully get front image.",
                "vision_states": vision_states
            }
        except Exception as e:
            logging.error(f"Get front image failed: {e}")
            logging.error(traceback.format_exc())
            return {
                "code": 500,
                "msg": "failed",
                "result": f"Failed to get front image. The reason is {e}",
                "vision_states": []
            }
