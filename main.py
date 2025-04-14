import os
import shutil
import json
import cv2
import numpy as np
import time
import re
import argparse
from models import VLM
from env import ThorEnvDogView
from agent import VLMNavigationAgent
from utils.video_utils import create_simulation_video
from utils.logger import logger

TASK_PROMT = """
You are a robot navigating a house. You see an image with numbered paths representing possible directions.
Your task is to follow these navigation instructions: '{TARGET}'
First, briefly describe what you see in the current view (e.g., "I see a kitchen with a counter and cabinets").
Then analyze the available paths ({ACTIONS}) and choose the best path number to follow the instructions.
If no path seems helpful, choose '0' to turn around.
Output your response as a JSON object with two keys: "reasoning" (your description and reasoning) and "action" (the chosen number as a string or '0').
Example: {{"reasoning": "I see a kitchen with a counter and cabinets. The instructions say to go left and then find the fridge. Path 3 leads to the left, which matches the first part of the instructions.", "action": "3"}}
Example: {{"reasoning": "I see a dead end with no clear paths forward. I should turn around to explore other directions.", "action": "0"}}
"""

def setup_views_directory():
    """Ensure the views directory is ready for storing images."""
    if not os.path.exists('views'):
        os.makedirs('views')
    else:
        shutil.rmtree('views')
        os.makedirs('views')

def depth_to_heatmap(depth):
    """Convert depth array to heatmap visualization"""
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

def save_step_data(step_number, view, depth, new_view, vlm_prompt, vlm_output_str, parsed_result):
    """Save all data for a simulation step to the views directory."""
    # Create step directory
    step_dir = os.path.join('views', f'step_{step_number}')
    os.makedirs(step_dir, exist_ok=True)
    
    # Save images
    cv2.imwrite(os.path.join(step_dir, 'augmented_view.jpg'), view)
    
    # Save depth as heatmap
    depth_heatmap = depth_to_heatmap(depth)
    cv2.imwrite(os.path.join(step_dir, 'depth.jpg'), depth_heatmap)
    
    # Save new view if available
    if new_view is not None:
        cv2.imwrite(os.path.join(step_dir, 'new_view.jpg'), new_view)
    
    # Save VLM prompt and output
    with open(os.path.join(step_dir, 'vlm_prompt.txt'), 'w') as f:
        f.write(vlm_prompt if vlm_prompt else "No prompt available")
    
    with open(os.path.join(step_dir, 'vlm_output.txt'), 'w') as f:
        f.write(vlm_output_str if vlm_output_str else "No output available")
    
    # Save parsed JSON result
    with open(os.path.join(step_dir, 'parsed_result.json'), 'w') as f:
        json.dump(parsed_result, f, indent=2)
    
    # Save a combined image for easier visualization
    if view is not None and depth is not None:
        # Get image dimensions
        height, width = view.shape[:2]
        
        # Create a side-by-side image
        combined = np.zeros((height, width * 2, 3), dtype=np.uint8)
        combined[:, :width] = view
        depth_heatmap_resized = cv2.resize(depth_heatmap, (width, height))
        combined[:, width:] = depth_heatmap_resized
        
        # Add text overlay with step number and action
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, f"Step {step_number+1}", (10, 30), font, 1, (255, 255, 255), 2)
        action = parsed_result.get('action', None)
        if action:
            cv2.putText(combined, f"Action: {action}", (10, 70), font, 1, (255, 255, 255), 2)
        
        # Save combined image
        cv2.imwrite(os.path.join(step_dir, 'combined.jpg'), combined)

def run_simulation(floor_id, model_id, api_url, target, task_prompt, max_steps, max_distance_to_move):
    """Run the navigation simulation and save results."""
    # Setup the views directory
    setup_views_directory()
    
    # Initialize the environment and agent
    print(f"Initializing simulation with floor {floor_id}...")
    env = ThorEnvDogView(floor_id)
    agent = VLMNavigationAgent(
        env=env,
        model_id=model_id,
        api_url=api_url,
        max_distance_to_move=max_distance_to_move
    )
    
    # Run simulation steps
    step_number = 0
    completed = False
    
    print(f"Starting simulation with target: {target}")
    print(f"Task prompt: {task_prompt}")
    
    # Display initial view if available
    if hasattr(agent, 'view') and agent.view is not None:
        initial_dir = os.path.join('views', 'initial')
        os.makedirs(initial_dir, exist_ok=True)
        cv2.imwrite(os.path.join(initial_dir, 'initial_view.jpg'), agent.view)
        if hasattr(agent, 'depth') and agent.depth is not None:
            depth_heatmap = depth_to_heatmap(agent.depth)
            cv2.imwrite(os.path.join(initial_dir, 'initial_depth.jpg'), depth_heatmap)
    
    while step_number < max_steps and not completed:
        print(f"\nExecuting step {step_number + 1}...")
        
        # Execute one step of the navigation
        augmented_view, actions_info, reasoning, action_chosen, new_view, is_completed = agent.step(
            target,
            task_prompt,
            max_steps
        )
        
        # Get the VLM prompt and output from agent
        vlm_prompt = agent.memory.get_last_vlm_prompt() if hasattr(agent.memory, 'get_last_vlm_prompt') else None
        vlm_output_str = agent.vlm_output_str
        
        # Parse the VLM output using agent's method
        reasoning, action = agent.parse_vlm_output(vlm_output_str)
        parsed_result = {"reasoning": reasoning, "action": action}
        
        # Save all data for this step
        save_step_data(
            step_number,
            augmented_view,
            agent.env.get_depth(),
            new_view,
            vlm_prompt,
            vlm_output_str,
            parsed_result
        )
        
        # Check if task is completed
        if is_completed:
            completed = True
            print(f"Task completed successfully at step {step_number + 1}!")
            break
            
        # Check if we have a valid action and view
        if augmented_view is None:
            print("Simulation ended: No valid view available")
            break
        
        # Print step information
        print(f"VLM chose action: {action_chosen}")
        print(f"Reasoning: {reasoning}")
        
        # Print any failed actions if available
        if hasattr(agent, 'failed_actions') and agent.failed_actions:
            for failed_action in agent.failed_actions:
                print(f"Failed action: {failed_action.action_number}, Reason: {failed_action.reason}")
        
        # Record completion check if available
        if hasattr(agent.memory, 'completion_checks') and agent.memory.completion_checks:
            latest_check = agent.memory.get_last_completion_check()
            if latest_check:
                print(f"Completion check: {latest_check.is_completed}")
                if latest_check.is_completed:
                    print(f"Completion reasoning: {latest_check.reasoning}")
        
        step_number += 1
                
    
    # Create simulation video
    print("Creating simulation video...")
    video_path = os.path.join('views', 'simulation_video.mp4')
    create_simulation_video(agent, video_path)
    print(f"Simulation video saved to {video_path}")
    
    # Final status
    if completed:
        print("Simulation completed successfully!")
    else:
        print(f"Simulation ended after {step_number} steps without completion.")
    
    return completed, step_number

def main():
    parser = argparse.ArgumentParser(description="VLM Navigation Simulation")
    parser.add_argument("--floor_id", type=str, default="FloorPlan_Train1_5", help="Floor ID for simulation")
    parser.add_argument("--model_id", type=str, default="Pro/Qwen/Qwen2.5-VL-7B-Instruct", help="Model ID for VLM")
    parser.add_argument("--api_url", type=str, default="http://10.8.25.28:8075/generate_action_proposals", help="API URL for VLM")
    parser.add_argument("--target", type=str, default="find TV", help="Target location")
    parser.add_argument("--max_steps", type=int, default=20, help="Maximum simulation steps")
    parser.add_argument("--max_distance", type=float, default=1.0, help="Maximum distance to move")
    
    args = parser.parse_args()
    
    # Print simulation parameters
    print("Starting simulation with the following parameters:")
    print(f"Floor ID: {args.floor_id}")
    print(f"Model ID: {args.model_id}")
    print(f"Target: {args.target}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Max Distance: {args.max_distance}")
    
    # Run the simulation
    completed, steps = run_simulation(
        floor_id=args.floor_id,
        model_id=args.model_id,
        api_url=args.api_url,
        target=args.target,
        task_prompt=TASK_PROMT,
        max_steps=args.max_steps,
        max_distance_to_move=args.max_distance,
    )
    
    # Print final statistics
    print("\nSimulation Summary:")
    print(f"Completed: {completed}")
    print(f"Total Steps: {steps}")
    
    # Create report file with summary
    with open(os.path.join('views', 'simulation_report.txt'), 'w') as f:
        f.write(f"Simulation Report\n")
        f.write(f"=================\n\n")
        f.write(f"Floor ID: {args.floor_id}\n")
        f.write(f"Model ID: {args.model_id}\n")
        f.write(f"Target: {args.target}\n")
        f.write(f"Completed: {completed}\n")
        f.write(f"Total Steps: {steps}\n")

if __name__ == "__main__":
    main()
