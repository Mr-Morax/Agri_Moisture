# Agri_Moisture
A streamlit and GEE project to help identify moisture requirements . using AI and Satellite data

STEP1.
git clone https://github.com/YOUR-USERNAME/Agri-Moisture.git
cd Agri-Moisture
STEP 2.
# Create the environment
python -m venv my_project_env

# Activate it (Linux/macOS)
source my_project_env/bin/bin/activate

# (Alternative for Windows users)
# .\my_project_env\Scripts\activate

STEP3.
pip install -r requirements.txt
STEP4.
earthengine authenticate
STEP5.
streamlit run app.py

note:Because you built that brilliant FakePkgResources polyfill directly into the top of your app.py, anyone running modern Python (like 3.12+) on any Linux distribution won't have to install setuptools manually. Your code will handle the deprecated pkg_resources missing module automatically at runtime!