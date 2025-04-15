# xvfb-run -a streamlit run streamlit_vlmnav_v2.py

# use debugpy on port 5678
xvfb-run -a debugpy --listen 5678 --wait-for-client main.py

xvfb-run python main.py --floor_id FloorPlan10 --model_id Pro/Qwen/Qwen2.5-VL-7B-Instruct --target "find a shelf with glass bottle on it" --max_steps 50 --max_distance 1.0

# xvfb-run python main.py --floor_id FloorPlan10 --model_id Pro/Qwen/Qwen2.5-VL-7B-Instruct --target "go left, then go stright, then move close to the shelf with glass bottle on it" --max_steps 50 --max_distance 1.0

# xvfb-run  -a debugpy --listen 5678 --wait-for-client main.py --floor_id FloorPlan_Train1_5 --model_id Pro/Qwen/Qwen2.5-VL-7B-Instruct --target "go straight, then turn left at first corner, go straight, then turn left at the first corner, then find the TV" --max_steps 20 --max_distance 1.0

# xvfb-run  python main.py --floor_id FloorPlan_Train1_5 --model_id Pro/Qwen/Qwen2.5-VL-7B-Instruct --target "find the spray" --max_steps 50 --max_distance 0.5


# ==================
xvfb-run -a debugpy --listen 5678 --wait-for-client main.py --floor_id FloorPlan_Train1_5 --action_model_id Qwen/Qwen2.5-VL-32B-Instruct --completion_model_id Pro/Qwen/Qwen2.5-VL-7B-Instruct --target "turn around, approach the baseball bat beside the bed " --max_steps 50 --max_distance 0.5
xvfb-run python main.py --floor_id FloorPlan_Train1_5 --action_model_id Qwen/Qwen2.5-VL-32B-Instruct --completion_model_id Pro/Qwen/Qwen2.5-VL-7B-Instruct --target "turn around, approach the baseball bat beside the bed " --max_steps 50 --max_distance 0.5
# ==================

xvfb-run python main.py --floor_id FloorPlan_Train1_5 --action_model_id Qwen/Qwen2.5-VL-32B-Instruct --completion_model_id Pro/Qwen/Qwen2.5-VL-7B-Instruct --target "turn around, approach the laptop on the table" --max_steps 50 --max_distance 0.5