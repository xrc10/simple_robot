from pathlib import Path
from typing import Any, Dict
import time
import requests
from urllib.parse import urljoin

from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry
from omagent_core.tool_system.base import ArgSchema, BaseTool

CURRENT_PATH = Path(__file__).parents[0]
SUFFIX = "/signalservice/robot/move"

ARGSCHEMA = {
    "vx": {
        "type": "number",
        "description": "Speed along the x-axis direction, in meters per second (m/s). Positive values indicate forward movement, and negative values indicate backward movement. If you don't know how far to go, you can try 1.5",
        "required": False,
    },
    "vy": {
        "type": "number",
        "description": "Speed along the y-axis direction, in meters per second (m/s). Positive values indicate movement to the left, and negative values indicate movement to the right.If you don't know how far to go, you can try 1.5",
        "required": False,
    },
    "vyaw": {
        "type": "number",
        "description": "Angular velocity around the z-axis, in radians per second (rad/s). Positive values indicate counterclockwise rotation, and negative values indicate clockwise rotation.",
        "required": False,
    },
}


@registry.register_tool()
class Move(BaseTool):
    """Tool for making Unitree Go2 robot move."""

    class Config:
        """Configuration for this pydantic object."""

        extra = "allow"
        arbitrary_types_allowed = True

    args_schema: ArgSchema = ArgSchema(**ARGSCHEMA)
    description: str = """This tool is used to control the robot dog to move. You can make the robot dog move forward, backward, left, right, and rotate by manipulating the vx, vy, and vyaw parameters."""
    request_url: str
    
    def _request_move(self, vx: float, vy: float, vyaw: float) -> Dict[str, Any]:
        url = urljoin(self.request_url, SUFFIX)
        response = requests.post(url, json={"vx": vx, "vy": vy, "vyaw": vyaw}).json()
        if response["code"] != '0':
            raise Exception(f"Robot move failed: {response['message']}")
        return response

    def _run(
        self,
        vx: float = 0,
        vy: float = 0,
        vyaw: float = 0 # vyaw=-1.5 means turn right for 45 degrees
    ) -> Dict[str, Any]:
        """Control the Go2 to move."""
        try:
            remaining_vx = abs(vx)
            remaining_vy = abs(vy)
            remaining_vyaw = abs(vyaw)
            
            vx_direction = 1 if vx > 0 else -1
            vy_direction = 1 if vy > 0 else -1
            vyaw_direction = 1 if vyaw > 0 else -1
            
            if remaining_vx > 3.8 or remaining_vy > 1.0 or remaining_vyaw > 4:
                while remaining_vx > 0 or remaining_vy > 0 or remaining_vyaw > 0:
                    current_vx = min(2, remaining_vx) * vx_direction if remaining_vx > 0 else 0
                    current_vy = min(1, remaining_vy) * vy_direction if remaining_vy > 0 else 0
                    current_vyaw = min(1.5, remaining_vyaw) * vyaw_direction if remaining_vyaw > 0 else 0
                    
                    code = self._request_move(current_vx, current_vy, current_vyaw)
                    if code != 0:
                        raise Exception(f"code: {code}")
                    
                    remaining_vx = max(0, remaining_vx - abs(current_vx))
                    remaining_vy = max(0, remaining_vy - abs(current_vy))
                    remaining_vyaw = max(0, remaining_vyaw - abs(current_vyaw))
                    
                    time.sleep(1)
            else:
                code = self._request_move(vx, vy, vyaw)
                
            if code != 0:
                raise Exception(f"code: {code}")
            result_string = "Successfully move the robot dog."
            if vx != 0:
                result_string += f"forward: {vx}m"
            if vy != 0:
                result_string += f"left: {vy}m"
            if vyaw != 0:
                result_string += f"rotate: {vyaw}rad"
            return {
                "code": code,
                "msg": "success",
                "result": result_string
            }
        except Exception as e:
            logging.error(f"Move failed: {e}")
            return {
                "code": 500,
                "msg": "failed",
                "result": f"Failed to move the robot dog. The reason is {e}"
            }
