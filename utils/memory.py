from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from utils.logger import logger

@dataclass
class ActionRecord:
    """Record of an action taken by the agent"""
    type: str  # 'action' or 'completion'
    step_number: int
    action_number: Optional[int] = None
    reasoning: str = ""
    vlm_prompt: str = ""
    vlm_response: str = ""
    error: Optional[str] = None

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
        logger.info(f"Added completion check: {check}")
    
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