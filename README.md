# simple_robot

A robot navigation system that uses vision-language models (VLMs) to make navigation decisions based on visual input. The system supports both simulated environments (AI2Thor) and real robot platforms (UnitreeDog).

## Description

This project implements a sophisticated robot navigation system that:
- Processes first-person view images using vision-language models
- Makes navigation decisions based on natural language instructions
- Supports both simulated environments (AI2Thor) and real robot platforms (UnitreeDog)
- Provides detailed logging and visualization of navigation steps
- Generates simulation videos and reports

## Requirements

- Python 3.10+
- AI2Thor (for simulated environment)
- OpenCV
- PyYAML
- PIL/Pillow
- NumPy
- Access to a VLM API endpoint (e.g., Qwen2.5-VL models)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/simple_robot.git
cd simple_robot

# Create and activate conda environment
conda create -p /data23/xu_ruochen/conda_envs/simple_robot python=3.10 -y
conda activate /data23/xu_ruochen/conda_envs/simple_robot

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `llm.yaml` file with your API credentials:

```yaml
SiliconCloud:
  api_key: "your_api_key_here"
  base_url: "your_base_url_here"
```

## Usage

The main program can be run with various command line arguments:

```bash
xvfb-run python main.py [options]
```

### Key Command Line Arguments

- `--floor_id`: Floor ID for simulation (default: "FloorPlan10")
- `--action_model_id`: Model ID for VLM (default: "Pro/Qwen/Qwen2.5-VL-7B-Instruct")
- `--completion_model_id`: Model ID for completion (default: "Qwen/Qwen2.5-VL-32B-Instruct")
- `--api_url`: API URL for VLM (default: "http://10.8.25.28:8075/generate_action_proposals")
- `--target`: Target location/instruction (default: "find a shelf with glass bottle on it")
- `--max_steps`: Maximum simulation steps (default: 50)
- `--max_distance`: Maximum distance to move (default: 1.0)
- `--env_type`: Environment type: "thor" or "unitree_dog" (default: "thor")
- `--dog_request_url`: API URL for UnitreeDog environment
- `--camera_fov`: Camera field of view for UnitreeDog (default: 80)

### Example Usage

1. Run in simulated environment:
```bash
xvfb-run python main.py --env_type thor --floor_id FloorPlan10 --target "find the kitchen table"
```

2. Run with real robot:
```bash
xvfb-run python main.py --env_type unitree_dog --dog_request_url "your_dog_api_url" --target "find the red chair"
```

## Output

The program generates:
- Step-by-step navigation logs in the `views` directory
- Visualizations of each step including:
  - Augmented views
  - Depth maps
  - Combined visualizations
- A simulation video (`simulation_video.mp4`)
- A detailed simulation report (`simulation_report.txt`)

## Project Structure

- `main.py`: Main program entry point
- `env.py`: Environment wrapper for AI2Thor
- `ut_env.py`: Environment wrapper for UnitreeDog
- `models.py`: Vision-language model implementation
- `agent.py`: Navigation agent implementation
- `utils/`: Utility functions for video creation and logging

## License

[Your license information here]
