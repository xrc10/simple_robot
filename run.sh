# xvfb-run -a streamlit run streamlit_vlmnav_v2.py

# use debugpy on port 5678
xvfb-run -a debugpy --listen 5678 --wait-for-client main.py