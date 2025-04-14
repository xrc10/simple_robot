import json
from typing import List, Dict, Any, Optional
import numpy as np
import json_repair

class NavigationMemory:
    def __init__(self):
        self.overall_goal: str = ""
        self.achieved_tasks: List[str] = []
        self.remaining_tasks: List[str] = []
        self.explored_areas: List[Dict[str, Any]] = []
        self.action_history: List[Dict[str, Any]] = []
        self.explored_areas_analysis: List[Dict[str, Any]] = []
        
    def update_goal(self, goal: str):
        """Update the overall goal of the navigation task"""
        self.overall_goal = goal
        
    def add_achieved_task(self, task: str):
        """Add a completed task to the memory"""
        if task not in self.achieved_tasks:
            self.achieved_tasks.append(task)
            if task in self.remaining_tasks:
                self.remaining_tasks.remove(task)
                
    def add_remaining_task(self, task: str):
        """Add a task that still needs to be completed"""
        if task not in self.remaining_tasks and task not in self.achieved_tasks:
            self.remaining_tasks.append(task)
            
    def update_explored_area(self, view: np.ndarray, depth: np.ndarray):
        """Record information about an explored area"""
        area_info = {
            "view": view.tolist(),  # Convert numpy array to list for JSON serialization
            "depth": depth.tolist(),
            "timestamp": len(self.explored_areas)
        }
        self.explored_areas.append(area_info)
        
    def add_action(self, action: Dict[str, Any]):
        """Record an action taken by the agent"""
        self.action_history.append(action)
        
    def is_position_visited(self, position: tuple, threshold: float = 0.5) -> bool:
        """Check if a position has been visited before, with some tolerance"""
        for visited_pos in self.visited_positions:
            if np.linalg.norm(np.array(position) - np.array(visited_pos)) < threshold:
                return True
        return False
        
    def get_memory_summary(self, n: int = 5) -> str:
        """Generate a summary of the current memory state"""
        summary = {
            "overall_goal": self.overall_goal,
            "achieved_tasks": self.achieved_tasks,
            "remaining_tasks": self.remaining_tasks,
            "explored_areas": self.explored_areas_analysis[-n:],
            "action_history": self.action_history[-n:]
        }
        return json.dumps(summary, indent=2)
        
    def get_recent_actions(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the n most recent actions"""
        return self.action_history[-n:] if self.action_history else []
        
    def is_repeating_pattern(self, pattern_length: int = 3) -> bool:
        """Check if the agent is stuck in a repeating pattern of actions"""
        if len(self.action_history) < pattern_length * 2:
            return False
            
        recent_actions = self.get_recent_actions(pattern_length * 2)
        first_half = recent_actions[:pattern_length]
        second_half = recent_actions[pattern_length:]
        
        return first_half == second_half
        
    def save_memory(self, filepath: str):
        """Save the memory state to a file"""
        memory_data = {
            "overall_goal": self.overall_goal,
            "achieved_tasks": self.achieved_tasks,
            "remaining_tasks": self.remaining_tasks,
            "action_history": self.action_history
        }
        with open(filepath, 'w') as f:
            json.dump(memory_data, f, indent=2)
            
    def load_memory(self, filepath: str):
        """Load the memory state from a file"""
        with open(filepath, 'r') as f:
            memory_data = json.load(f)
            self.overall_goal = memory_data["overall_goal"]
            self.achieved_tasks = memory_data["achieved_tasks"]
            self.remaining_tasks = memory_data["remaining_tasks"]
            self.action_history = memory_data["action_history"]
            
    def get_memory_analysis_prompt(self, target: str) -> str:
        """Generate the prompt for memory analysis"""
        return f"""
        Analyze the current navigation state and update the memory accordingly.
        
        Current Navigation Instructions: {target}
        
        Current Memory State:
        {self.get_memory_summary()}
        
        Recent Actions:
        {self.get_recent_actions()}
        
        Please analyze the current situation and provide a JSON response with:
        1. overall_goal: The main objective of the navigation task (should be related to following the instructions: {target})
        2. achieved_tasks: List of steps from the instructions that have been completed
        3. remaining_tasks: List of steps from the instructions that still need to be completed
        4. explored_areas_analysis: Analysis of what has been seen in explored areas and how it relates to following the instructions
        
        Format your response as a JSON object.
        """
    
    def add_explored_area(self, analysis: Dict[str, Any]):
        """Add an explored area to the memory"""
        self.explored_areas_analysis.append(analysis)

    def update_from_vlm_analysis(self, vlm_output: str):
        """Update memory state based on VLM analysis of the current situation"""
        try:
            if "```json" in vlm_output:
                vlm_output = vlm_output.split("```json")[1].split("```")[0]
            elif "```" in vlm_output:
                vlm_output = vlm_output.split("```")[1]
            else:
                import re
                match = re.match(r"\{(.*)\}", vlm_output)
                if match:
                    vlm_output = '{' + match.group(1) + '}'
            # Parse the VLM output which should be in JSON format
            analysis = json_repair.loads(vlm_output)
            
            print(analysis)

            # Update overall goal if provided
            if "overall_goal" in analysis:
                self.overall_goal = analysis["overall_goal"]
                
            # Update achieved tasks
            if "achieved_tasks" in analysis:
                for task in analysis["achieved_tasks"]:
                    self.add_achieved_task(task)
                    
            # Update remaining tasks
            if "remaining_tasks" in analysis:
                for task in analysis["remaining_tasks"]:
                    self.add_remaining_task(task)
                    
            # Update explored areas analysis if provided
            if "explored_areas_analysis" in analysis:
                # This could include information about what has been seen in each area
                self.add_explored_area(analysis["explored_areas_analysis"])
                
        except json.JSONDecodeError:
            print("Failed to parse VLM memory analysis output")
