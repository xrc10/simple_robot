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
from utils.occupancy_map import generate_occupancy_map

TASK_PROMPT = """
You are a robot navigating a house. You see an image with numbered paths representing possible directions.
Your task is to follow these navigation instructions: '{TARGET}'

ANALYZE YOUR SURROUNDINGS:
- Describe what you see in your current view.
- Note any key landmarks or objects that are clearly visible to you in your current view.
- IMPORTANT: Only report landmarks that you can clearly see and identify in your current view.
- DO NOT report landmarks that are:
  * Partially visible
  * Uncertain
  * Seen through a path/doorway
  * Previously seen but not in current view
  * Hallucinated or assumed

UNDERSTAND YOUR TASK:
- For simple tasks like "find X": Focus on identifying the target object.
- For multi-step directions: Break the instruction into logical steps.
- Track your progress on completed steps.

PLAN YOUR NEXT MOVE:
- Choose a path from {ACTIONS} that best advances your goal.
- If you're uncertain or lost, choose '0' to turn around.
- Avoid previously failed paths or detected loops.

MEMORY OF PREVIOUS STEPS:
{MEMORY}

DETERMINE TASK COMPLETION:
- After each action execution, determine if you've completed the task.
- For "find X" tasks, you've completed the task when X is clearly visible in your current view.
- For multi-step directions, you've completed the task when all steps are done.
- Consider task complete only when you're confident you've met all requirements.

Example for simple task: {{"reasoning": "The instruction is "find the TV". I see what appears to be a TV on a stand through path 2. Since my task is to find the TV, I should go that way.", "action": "2", "progress": "TV spotted, moving toward it", "landmarks": "black TV, brown couch, yellow coffee table", "completed": false}}

Example for complex task: {{"reasoning": "The instruction is "go straight in the hallway until the end, then turn left, then find the TV". Given previous steps, I am still in the hallway. I should go straight until the end. Path 1 leads straight ahead which matches the first part of my instructions.", "action": "1", "progress": "Follwoing first part of the instruction to go straight in the hallway", "landmarks": "Hallway, red picture frames", "completed": false}}

Example for completed task: {{"reasoning": "The instruction is "find the TV". I can now clearly see a TV directly in front of me. The task was to find the TV and I have successfully found it.", "action": "done", "progress": "Found the TV", "landmarks": "Black flat-screen TV, wooden TV stand, potted plant", "completed": true}}

Output your response as a JSON object with these keys:
"reasoning": Your analysis of the current situation and plan
"action": The chosen path number as a string (e.g., "1", "2", "3"), "0" to turn around, or "done" if task is complete
"progress": Brief description of your task progress (e.g., "Found kitchen, looking for TV")
"landmarks": ONLY key objects or features that are clearly visible in your current view. Do not include uncertain or partially visible objects.
"completed": Boolean (true/false) indicating whether the task has been completed
"""

class VLMNavigationAgent:
    """Agent for VLM-based navigation"""
    def __init__(self, env: ThorEnvDogView, action_model_id: str, completion_model_id: str, api_url: str, max_distance_to_move: float = 1.0):
        """Initialize the VLMNavigationAgent
        
        Args:
            env: The Thor environment
            action_model_id: ID of the VLM model to use for action selection
            completion_model_id: ID of the model for completion checks (kept for backward compatibility, not used anymore)
            api_url: URL of the API for action proposals
            max_distance_to_move: Maximum distance to move in meters
        """
        self.env = env
        self.action_model = VLM('llm.yaml', action_model_id)
        # We no longer need a separate completion model as completion is determined by the action model
        self.api_url = api_url
        self.max_distance_to_move = max_distance_to_move
        self.memory = NavigationMemory()
        self.step_number = 0
        self.completed = False
        self.vlm_output_str = ""
        self.last_action_was_turn_around = False  # Track if the last action was a turn around

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
        self.last_n_for_action_choice = 5
        self.last_n_for_completion_check = 3  # Number of recent actions/images to consider for task completion check
        
        # Occupancy map parameters
        self.camera_height = 1.5  # meters
        self.camera_fov = self.env.camera_fov    # FOV degrees
        self.camera_pitch = 30  # degrees down from horizontal
        
        self.setup_views_directory()

    def setup_views_directory(self):
        """Ensure the views directory is ready for storing images."""
        if not os.path.exists('views'):
            os.makedirs('views')
        else:
            shutil.rmtree('views')
            os.makedirs('views')
        
        # Create directory for occupancy maps
        occupancy_dir = os.path.join('views', 'occupancy_maps')
        if not os.path.exists(occupancy_dir):
            os.makedirs(occupancy_dir)

    def get_action_proposals(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Get action proposals from the API"""
        image_base64 = convert_image_to_base64(image)
        response = send_api_request(image_base64, self.api_url, self.min_angle, self.number_size, self.min_path_length, self.min_arrow_width)
        return handle_api_response(response)

    def parse_vlm_output(self, vlm_output_str):
        """Parse the VLM output to extract reasoning and action."""
        reasoning = "Could not parse VLM output."
        action_chosen = None
        progress = ""
        landmarks = ""
        completed = False
        
        try:
            json_pattern = re.compile(r"```json\s*({.*?})\s*```|({.*?})", re.DOTALL)
            match = json_pattern.search(vlm_output_str)
            if match:
                json_str = match.group(1) or match.group(2)
                if json_str:
                    vlm_output_json = json_repair.loads(json_str)
                    reasoning = vlm_output_json.get("reasoning", "No reasoning provided in JSON.")
                    action_chosen = vlm_output_json.get("action", None)
                    progress = vlm_output_json.get("progress", "")
                    landmarks = vlm_output_json.get("landmarks", "")
                    completed = vlm_output_json.get("completed", False)
            else:
                vlm_output_json = json_repair.loads(vlm_output_str.strip())
                reasoning = vlm_output_json.get("reasoning", "No reasoning provided.")
                action_chosen = vlm_output_json.get("action", None)
                progress = vlm_output_json.get("progress", "")
                landmarks = vlm_output_json.get("landmarks", "")
                completed = vlm_output_json.get("completed", False)
        except json.JSONDecodeError:
            raise Exception("Failed to parse VLM output as JSON.")
        return reasoning, action_chosen, progress, landmarks, completed

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

            # update self.memory with the action
            self.memory.get_last_action().movement_info = {
                "action": "TurnAround",
                "degrees": 180
            }
            return event
        
        # Calculate move distance based on depth and boundary point
        move_distance = self.max_distance_to_move
        depth = self.env.get_depth()
        if action_info["boundary_point"] is not None and depth is not None:
            boundary_x, boundary_y = action_info["boundary_point"]
            boundary_distance = depth[boundary_y, boundary_x]  # Get depth in meters
            move_distance = min(1/2 * boundary_distance, self.max_distance_to_move)
        
        # Otherwise, rotate to the specific degree and move forward
        rotation_action = "RotateLeft" if degree < 0 else "RotateRight"
        event = self.env.step(rotation_action, degrees=abs(degree))
        event = self.env.step("MoveAhead", magnitude=move_distance)

        if event.metadata["lastActionSuccess"]:
            logger.info(f"Action {rotation_action} with degree {abs(degree)} and move distance {move_distance} succeeded")
        else:
            failed_reason = event.metadata['errorMessage']
            logger.info(f"Action {rotation_action} with degree {abs(degree)} and move distance {move_distance} failed")
            logger.info(f"Failed reason: {failed_reason}")
            raise Exception(f"Action {rotation_action} with degree {abs(degree)} and move distance {move_distance} failed")

        # update self.memory with the action
        self.memory.get_last_action().movement_info = {
            "action": rotation_action,
            "degrees": abs(degree),
            "move_distance": move_distance
        }

        return event
    
    def step(self, target: str, max_steps: int) -> Dict[str, Any]:
        """Execute one step of navigation"""
        result = {
            "augmented_view": None,
            "current_view": None,
            "depth": None,
            "actions_info": None,
            "reasoning": "",
            "action_chosen": "",
            "new_view": None,
            "completed": False,
            "vlm_prompt": "",
            "vlm_output_str": "",
            "completion_reasoning": "",
            "completion_prompt": "",  # For backward compatibility
            "completion_output": "",  # For backward compatibility
            "completion_parsed_result": None,  # For backward compatibility
            "failed_action": None,
            "progress": "",
            "landmarks": ""
        }
        
        if self.completed or self.step_number >= max_steps:
            result["completed"] = self.completed
            return result
        
        try:
            # Get current view and depth
            current_view = self.env.get_observation()
            depth = self.env.get_depth()
            result["current_view"] = current_view
            result["depth"] = depth
            
            # Get action proposals
            augmented_view, actions_info, navigability_mask = self.get_action_proposals(current_view)
            result["augmented_view"] = augmented_view
            result["actions_info"] = actions_info
            result["navigability_mask"] = navigability_mask
            
            # Format actions for prompt
            actions_str = ", ".join([f"{a['action_number']}" for a in actions_info])
            memory_str = self.memory.get_action_history_as_string(self.last_n_for_action_choice)
            formatted_prompt = TASK_PROMPT.format(TARGET=target, ACTIONS=actions_str, MEMORY=memory_str)
            result["vlm_prompt"] = formatted_prompt
            
            # Get VLM response
            self.vlm_output_str = self.action_model.get_response(augmented_view, formatted_prompt)
            result["vlm_output_str"] = self.vlm_output_str
            
            reasoning, action_chosen, progress, landmarks, completed = self.parse_vlm_output(self.vlm_output_str)
            result["reasoning"] = reasoning
            result["action_chosen"] = action_chosen
            result["progress"] = progress
            result["landmarks"] = landmarks
            result["completed"] = completed
            result["completion_reasoning"] = reasoning  # Use the same reasoning for completion
            
            # Handle consecutive turn around constraint
            if action_chosen and action_chosen != "done":
                action_number = int(action_chosen)
                
                # Check if this is a second consecutive turn around
                if action_number == 0 and self.last_action_was_turn_around:
                    # Get other available actions
                    other_actions = [a for a in actions_info if a["action_number"] != 0]
                    
                    if other_actions:
                        # Randomly choose another action
                        import random
                        random_action = random.choice(other_actions)
                        action_number = random_action["action_number"]
                        action_chosen = str(action_number)
                        reasoning += f" [OVERRIDE: Avoiding consecutive turn arounds. Randomly chose action {action_number} instead.]"
                    else:
                        # No other actions available, keep the turn around
                        reasoning += " [NOTE: Forced to turn around again as no other actions available.]"
                
                # Record if this action is a turn around for next time
                self.last_action_was_turn_around = (action_number == 0)
                result["action_chosen"] = action_chosen
                result["reasoning"] = reasoning
            
            # Record action
            action_record = ActionRecord(
                step_number=self.step_number,
                action_number=int(action_chosen) if action_chosen and action_chosen != "done" else None,
                reasoning=reasoning,
                vlm_prompt=formatted_prompt,
                vlm_response=self.vlm_output_str,
                progress=progress,
                landmarks=landmarks,
                completed=completed
            )
            self.memory.add_action(action_record)
            
            # Draw action overlay
            if action_chosen and action_chosen != "done":
                action_number = int(action_chosen)
                action_info = next((a for a in actions_info if a["action_number"] == action_number), None)
                augmented_view = draw_action_overlay(augmented_view, action_info)
                result["augmented_view"] = augmented_view

            # Store images, now the augmented_view has draw action overlay (if action is chosen)
            self.memory.add_images(augmented_view, current_view, depth, navigability_mask)
            
            # Generate occupancy map
            occupancy_map_path = os.path.join('views', 'occupancy_maps', f'occupancy_map_step_{self.step_number}.png')
            generate_occupancy_map(
                memory=self.memory,
                camera_height=self.camera_height,
                camera_fov=self.camera_fov,
                camera_pitch=self.camera_pitch,
                display=False,
                save_path=occupancy_map_path
            )
            
            # Execute action and get new view
            if action_chosen and action_chosen != "done":
                try:
                    event = self.execute_action(int(action_chosen), actions_info)
                    new_view = event.frame
                    result["new_view"] = new_view
                    self.memory.get_last_action().is_success = True
                except Exception as e:
                    logger.error(f"Failed to execute action: {str(e)}")
                    failed_action = {
                        "step": self.step_number,
                        "type": "action_execution",
                        "error": str(e),
                        "action_number": action_chosen,
                        "vlm_prompt": formatted_prompt,
                        "vlm_response": self.vlm_output_str
                    }
                    self.memory.add_failed_action(failed_action)
                    result["failed_action"] = failed_action
            
            # Set backward compatibility fields for completion check
            result["completion_prompt"] = formatted_prompt
            result["completion_output"] = self.vlm_output_str
            result["completion_parsed_result"] = {
                "completed": completed,
                "reasoning": reasoning
            }
            
            # Set task completion based on VLM's determination
            if completed:
                self.completed = True
                
                # Create a completion check record for compatibility with existing code
                check = CompletionCheck(
                    step_number=self.step_number,
                    completed=completed,
                    reasoning=reasoning,
                    images=[current_view],
                    prompt=formatted_prompt,
                    response=self.vlm_output_str
                )
                self.memory.add_completion_check(check)
                
                # Generate final occupancy map
                occupancy_map_path = os.path.join('views', 'occupancy_maps', 'final_occupancy_map.png')
                generate_occupancy_map(
                    memory=self.memory,
                    camera_height=self.camera_height,
                    camera_fov=self.camera_fov,
                    camera_pitch=self.camera_pitch,
                    display=False,
                    save_path=occupancy_map_path
                )
            
            self.step_number += 1
            return result
            
        except Exception as e:
            logger.error(f"Error during step: {str(e)}")
            failed_action = {
                "step": self.step_number,
                "type": "step_execution",
                "error": str(e)
            }
            self.memory.add_failed_action(failed_action)
            result["failed_action"] = failed_action
            result["reasoning"] = str(e)
            return result
