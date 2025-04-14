from typing import Tuple, Optional, Dict, Any, List
import cv2
import os
import shutil
import base64
import requests
import json
import numpy as np
import re
import json_repair
from models import VLM
from env import ThorEnvDogView
from utils.memory import NavigationMemory, ActionRecord, CompletionCheck
from utils.api_utils import send_api_request, handle_api_response
from utils.image_utils import convert_image_to_base64
from utils.visualization import draw_action_overlay
from utils.logger import logger

class VLMNavigationAgent:
    """Agent for VLM-based navigation"""
    def __init__(self, env: ThorEnvDogView, model_id: str, api_url: str, max_distance_to_move: float = 1.0):
        self.env = env
        self.model = VLM('llm.yaml', model_id)
        self.api_url = api_url
        self.max_distance_to_move = max_distance_to_move
        self.memory = NavigationMemory()
        self.step_number = 0
        self.completed = False
        self.vlm_output_str = ""

        # for action proposal API
        self.min_angle = 20
        self.number_size = 30
        self.min_path_length = 80
        self.min_arrow_width = 60
        
        # Properties for easy access to memory
        self.action_memory = self.memory.action_memory
        self.augmented_images = self.memory.augmented_images
        self.complete_images = self.memory.complete_images
        self.depth_memory = self.memory.depth_memory
        self.failed_actions = self.memory.failed_actions
        self.view = None
        self.depth = None
        self.last_actions_info = None
        self.last_n_for_summary = 5
        self.last_n_for_done = 3  # Number of recent actions/images to consider for task completion check
        self.setup_views_directory()

    def setup_views_directory(self):
        """Ensure the views directory is ready for storing images."""
        if not os.path.exists('views'):
            os.makedirs('views')
        else:
            shutil.rmtree('views')
            os.makedirs('views')

    def get_action_proposals(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Get action proposals from the API"""
        image_base64 = convert_image_to_base64(image)
        response = send_api_request(image_base64, self.api_url, self.min_angle, self.number_size, self.min_path_length, self.min_arrow_width)
        return handle_api_response(response)

    def parse_vlm_output(self, vlm_output_str):
        """Parse the VLM output to extract reasoning and action."""
        reasoning = "Could not parse VLM output."
        action_chosen = None
        try:
            json_pattern = re.compile(r"```json\s*({.*?})\s*```|({.*?})", re.DOTALL)
            match = json_pattern.search(vlm_output_str)
            if match:
                json_str = match.group(1) or match.group(2)
                if json_str:
                    vlm_output_json = json_repair.loads(json_str)
                    reasoning = vlm_output_json.get("reasoning", "No reasoning provided in JSON.")
                    action_chosen = vlm_output_json.get("action", None)
            else:
                vlm_output_json = json_repair.loads(vlm_output_str.strip())
                reasoning = vlm_output_json.get("reasoning", "No reasoning provided.")
                action_chosen = vlm_output_json.get("action", None)
        except json.JSONDecodeError:
            raise Exception("Failed to parse VLM output as JSON.")
        return reasoning, action_chosen

    def execute_action(self, action_number: int, actions_info: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute the chosen action"""
        # Find the action info for the chosen action number
        action_info = next((a for a in actions_info if a["action_number"] == action_number), None)
        
        if action_info is None:
            raise Exception(f"Error: Action {action_number} not found in action proposals")
        
        degree = action_info["turning_degree"]
        logger.info(f"Executing action {action_number} with degree {degree}")
        
        # If action is 0, turn around 180 degrees
        if action_number == 0:
            event = self.env.step("RotateRight", degrees=180)
            return event
        
        # Otherwise, rotate to the specific degree and move forward
        rotation_action = "RotateLeft" if degree < 0 else "RotateRight"
        event = self.env.step(rotation_action, degrees=abs(degree))
        event = self.env.step("MoveAhead", magnitude=self.max_distance_to_move)
        return event

    def update_memory_with_action(self, action_number, actions_info, reasoning):
        """Update memory with the current action and state"""
        
        # Update explored area
        self.memory.update_explored_area(self.view, self.depth)
        
        # Find the action info for the chosen action number
        action_info = next((a for a in actions_info if a["action_number"] == action_number), None)
        if action_info is None:
            return
            
        # Calculate move distance if applicable
        move_distance = None
        if action_info["boundary_point"] is not None and self.depth is not None:
            boundary_x, boundary_y = action_info["boundary_point"]
            boundary_distance = self.depth[boundary_y, boundary_x]  # Get depth in meters
            move_distance = min(2/3 * boundary_distance, self.max_distance_to_move)
            
        # Get turning degree
        turning_degree = 180 if action_number == 0 else action_info["turning_degree"]
        
        # Record the action with meaningful movement information
        action_record = {
            "turning_degree": turning_degree,
            "move_distance": move_distance,
            "reasoning": reasoning,
            "step_number": self.step_number
        }
        self.memory.add_action(action_record)
        
        # Check for repeating patterns
        if self.memory.is_repeating_pattern():
            print("Warning: Agent detected in repeating pattern of actions")

    def check_task_completion(self, target: str, current_view: np.ndarray) -> Tuple[bool, str]:
        """Check if the navigation task is complete"""
        completion_prompt = f"""You are a robot navigating a house. Your task is to find: {target}
        Look at the current view and determine if you have completed the task.
        Output your response as a JSON object with two keys:
        - "completed" (boolean): true if the task is complete, false otherwise
        - "reasoning" (string): explain why you think the task is complete or not
        """
        
        completion_check = self.model.get_response(current_view, completion_prompt)
        try:
            # use json_repair to parse the completion check
            completion_data = json_repair.loads(completion_check)
            completed = completion_data.get("completed", False)
            reasoning = completion_data.get("reasoning", "No reasoning provided")
            
            # Record completion check
            check = CompletionCheck(
                step_number=self.step_number,
                completed=completed,
                reasoning=reasoning,
                images=[current_view],
                prompt=completion_prompt,
                response=completion_check
            )
            self.memory.add_completion_check(check)
            
            return completed, reasoning
        except json.JSONDecodeError:
            logger.error("Failed to parse completion check response")
            return False, "Failed to parse completion check response"

    def step(self, target: str, task_prompt: str, max_steps: int) -> Tuple[Optional[np.ndarray], Optional[List[Dict[str, Any]]], str, str, Optional[np.ndarray], bool]:
        """Execute one step of navigation"""
        if self.completed or self.step_number >= max_steps:
            return None, None, "", "", None, self.completed
        
        try:
            # Get current view and depth
            current_view = self.env.get_observation()
            depth = self.env.get_depth()
            
            # Get action proposals
            augmented_view, actions_info = self.get_action_proposals(current_view)
            
            # Format actions for prompt
            actions_str = ", ".join([f"{a['action_number']}" for a in actions_info])
            formatted_prompt = task_prompt.format(TARGET=target, ACTIONS=actions_str)
            
            # Get VLM response
            self.vlm_output_str = self.model.get_response(augmented_view, formatted_prompt)
            reasoning, action_chosen = self.parse_vlm_output(self.vlm_output_str)
            
            # Record action
            action_record = ActionRecord(
                type="action",
                step_number=self.step_number,
                action_number=int(action_chosen) if action_chosen and action_chosen != "done" else None,
                reasoning=reasoning,
                vlm_prompt=formatted_prompt,
                vlm_response=self.vlm_output_str
            )
            self.memory.add_action(action_record)
            
            # Draw action overlay
            if action_chosen and action_chosen != "done":
                action_number = int(action_chosen)
                action_info = next((a for a in actions_info if a["action_number"] == action_number), None)
                augmented_view = draw_action_overlay(augmented_view, action_info)

            # Store images, now the agumented_view has draw action overlay (if action is chosen)
            self.memory.add_images(augmented_view, current_view, depth)
            
            # Execute action and get new view
            new_view = None
            if action_chosen and action_chosen != "done":
                try:
                    event = self.execute_action(int(action_chosen), actions_info)
                    new_view = event.frame
                except Exception as e:
                    logger.error(f"Failed to execute action: {str(e)}")
                    self.memory.add_failed_action({
                        "step": self.step_number,
                        "type": "action_execution",
                        "error": str(e),
                        "action_number": action_chosen,
                        "vlm_prompt": formatted_prompt,
                        "vlm_response": self.vlm_output_str
                    })
            
            # Check task completion
            completed, completion_reasoning = self.check_task_completion(target, current_view)
            if completed:
                self.completed = True
                completion_record = ActionRecord(
                    type="completion",
                    step_number=self.step_number,
                    reasoning=completion_reasoning
                )
                self.memory.add_action(completion_record)
            
            self.step_number += 1
            return augmented_view, actions_info, reasoning, action_chosen, new_view, completed
            
        except Exception as e:
            logger.error(f"Error during step: {str(e)}")
            self.memory.add_failed_action({
                "step": self.step_number,
                "type": "step_execution",
                "error": str(e)
            })
            return None, None, str(e), None, None, False
