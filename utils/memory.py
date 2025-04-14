from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from utils.logger import logger

@dataclass
class ActionRecord:
    """Record of an action taken by the agent"""
    step_number: int
    action_number: Optional[int] = None
    movement_info: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    vlm_prompt: str = ""
    vlm_response: str = ""
    error: Optional[str] = None
    is_success: bool = False
    progress: str = ""
    landmarks: str = ""

@dataclass
class CompletionCheck:
    """Record of a task completion check"""
    step_number: int
    completed: bool
    reasoning: str
    images: List[np.ndarray]
    prompt: str
    response: str

@dataclass
class NavigationMemory:
    """Memory management for navigation agent"""
    action_memory: List[ActionRecord] = field(default_factory=list)
    completion_checks: List[CompletionCheck] = field(default_factory=list)
    augmented_images: List[np.ndarray] = field(default_factory=list)
    complete_images: List[np.ndarray] = field(default_factory=list)
    depth_memory: List[np.ndarray] = field(default_factory=list)
    failed_actions: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_action(self, action_record: ActionRecord):
        """Add an action to memory"""
        self.action_memory.append(action_record)
        logger.info(f"Added action record: {action_record}")
    
    def add_completion_check(self, check: CompletionCheck):
        """Add a completion check to memory"""
        self.completion_checks.append(check)
        check_without_images = check.__dict__
        check_without_images.pop('images')
        logger.info(f"Added completion check: {check_without_images}")
    
    def add_images(self, augmented: np.ndarray, complete: np.ndarray, depth: np.ndarray):
        """Add images to memory"""
        self.augmented_images.append(augmented)
        self.complete_images.append(complete)
        self.depth_memory.append(depth)
        logger.debug(f"Added images to memory - shapes: augmented={augmented.shape}, complete={complete.shape}, depth={depth.shape}")
    
    def add_failed_action(self, failed_action: Dict[str, Any]):
        """Add a failed action to memory"""
        self.failed_actions.append(failed_action)
        logger.warning(f"Added failed action: {failed_action}")
    
    def get_last_action(self) -> Optional[ActionRecord]:
        """Get the last action taken"""
        return self.action_memory[-1] if self.action_memory else None
    
    def get_last_completion_check(self) -> Optional[CompletionCheck]:
        """Get the last completion check"""
        return self.completion_checks[-1] if self.completion_checks else None

    def clear(self):
        """Clear all memory"""
        self.action_memory.clear()
        self.completion_checks.clear()
        self.augmented_images.clear()
        self.complete_images.clear()
        self.depth_memory.clear()
        self.failed_actions.clear()
        logger.info("Cleared all memory") 

    def get_action_history_as_string(self, last_n_actions: int = 5) -> str:
        """Get the action memory history as a string"""
        if not self.action_memory:
            return "No actions taken yet."
        
        # Get the last n actions
        relevant_actions = self.action_memory[-last_n_actions:] if len(self.action_memory) > last_n_actions else self.action_memory
        
        # Format each action into a string
        action_strings = []
        for action in relevant_actions:
            action_string = f"- Step {action.step_number}: "
            
            # Add goal information
            if action.progress:
                action_string += f"Progress: {action.progress} | "
            
            # Add reasoning summary (limited to keep memory concise)
            if action.reasoning:
                # Truncate reasoning to keep it concise
                max_reasoning_len = 200
                reasoning_summary = action.reasoning[:max_reasoning_len] + "..." if len(action.reasoning) > max_reasoning_len else action.reasoning
                action_string += f"Reasoning: {reasoning_summary} | "
            
            # Add movement information
            if action.movement_info:
                action_string += f"Action: {action.movement_info.get('action', 'Unknown')} "
                if 'degrees' in action.movement_info:
                    action_string += f"({action.movement_info['degrees']} degrees) "
                if 'move_distance' in action.movement_info:
                    action_string += f"({action.movement_info['move_distance']} meters) "
                action_string += f"| Success: {'Yes' if action.is_success else 'No'} | "
            
            # Add landmarks information
            if action.landmarks:
                action_string += f"Landmarks: {action.landmarks}"
            
            action_strings.append(action_string.rstrip(" | "))  # Remove trailing separator
        
        # Add a summary of overall progress
        all_landmarks = set()
        
        for action in self.action_memory:
            # If landmarks is a string, split by commas and strip whitespace
            if action.landmarks and isinstance(action.landmarks, str):
                landmark_items = [item.strip() for item in action.landmarks.split(',')]
                all_landmarks.update(landmark_items)
        
        summary = []
        if all_landmarks:
            summary.append(f"LANDMARKS DISCOVERED: {', '.join(all_landmarks)}")
        
        # Add the summary at the beginning
        if summary:
            action_strings = summary + [""] + action_strings
        
        return "\n".join(action_strings)
