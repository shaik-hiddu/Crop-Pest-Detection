import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json
import os
from huggingface_hub import hf_hub_download
import urllib.request # Import for handling URLs

# ========== Constants ==========
USER_FILE = "users.json"
MODEL_PATH = "final_model.h5"
IMG_SIZE = (380, 380)
# PESTICIDE_IMAGE_DIR = "assets/pesticide_images" # No need to define, we handle dynamically

# ========== Authentication ==========
def load_users():
    if not os.path.exists(USER_FILE):
        with open(USER_FILE, "w") as f:
            json.dump({}, f)
    with open(USER_FILE, "r") as f:
        return json.load(f)

def save_user(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = password
    with open(USER_FILE, "w") as f:
        json.dump(users, f)
    return True

def authenticate(username, password):
    users = load_users()
    return users.get(username) == password

# ========== Load Model ==========
@st.cache_resource
def load_pest_model():
    model_path = hf_hub_download(repo_id="hiddu2004/hello", filename="final_model2.h5")
    return load_model(model_path)

model = load_pest_model()

# ========== Class Labels ==========
# Modified to use URLs instead of local file paths.  This is CRUCIAL
# because the original code assumes the 'assets/pesticide_images' directory
# exists locally, which might not be the case when running in a cloud environment.
class_labels = {
    0: {'pest': 'aphid', 'pesticide': 'Imidacloprid', 'image': 'https://github.com/shaik-hiddu/Crop-Pest-Detection/assets/pesticide_images/aphid_pesticide.jpg'}, # Replace with actual URL
    1: {'pest': 'armyworm', 'pesticide': 'Lambda-cyhalothrin', 'image': 'https://github.com/shaik-hiddu/Crop-Pest-Detection/assets/pesticide_images/armyworm_pesticide.jpg'}, # Replace with actual URL
    2: {'pest': 'beetle', 'pesticide': 'Carbaryl', 'image': 'https://github.com/shaik-hiddu/Crop-Pest-Detection/assets/pesticide_images/beetle_pesticide.jpeg'},   # Replace with actual URL
    3: {'pest': 'bollworm', 'pesticide': 'Chlorpyrifos', 'image': 'https://github.com/shaik-hiddu/Crop-Pest-Detection/assets/pesticide_images/bollworm_pesticide.jpeg'}, # Replace with actual URL
    4: {'pest': 'grasshopper', 'pesticide': 'Malathion', 'image': 'https://github.com/shaik-hiddu/Crop-Pest-Detection/assets/pesticide_images/grasshopper_pesticide.jpeg'}, # Replace with actual URL
    5: {'pest': 'mites', 'pesticide': 'Abamectin', 'image': 'https://github.com/shaik-hiddu/Crop-Pest-Detection/assets/pesticide_images/mites_pesticide.jpg'},       # Replace with actual URL
    6: {'pest': 'mosquito', 'pesticide': 'Temephos', 'image': 'https://github.com/shaik-hiddu/Crop-Pest-Detection/assets/pesticide_images/mosquito_pesticide.jpeg'}, # Replace with actual URL
    7: {'pest': 'sawfly', 'pesticide': 'Spinosad', 'image': 'https://github.com/shaik-hiddu/Crop-Pest-Detection/assets/pesticide_images/sawfly_pesticide.jpeg'},     # Replace with actual URL
    8: {'pest': 'stem_borer', 'pesticide': 'Quinalphos', 'image': 'https://github.com/shaik-hiddu/Crop-Pest-Detection/assets/pesticide_images/stem_borer_pesticide.jpg'}   # Replace with actual URL
}

# ========== Session State ==========
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ========== Login/Register Page ==========
def login_page():
    st.title("🔐 Login / Register")
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
    
    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate(username, password):
                st.success(f"Welcome back, {username}!")
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password")

    with tab2:
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        if st.button("Register"):
            if save_user(new_user, new_pass):
                st.success("User registered! You can now log in.")
            else:
                st.warning("Username already exists.")

# ========== Pest Detection Page ==========
def pest_detection_page():
    st.title("🌿 Crop Pest Detector")
    st.write(f"👤 Logged in as **{st.session_state.username}**")
    
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

    uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image_data = Image.open(uploaded_file).convert("RGB")
        st.image(image_data, caption="Uploaded Image", use_column_width=True)

        # Preprocessing the image for the model
        img = image_data.resize(IMG_SIZE)
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Prediction
        prediction = model.predict(img)
        class_idx = np.argmax(prediction, axis=1)[0]
        pest_info = class_labels.get(class_idx)

        if pest_info:
            st.success(f"🪲 **Detected Pest:** {pest_info['pest']}")
            st.markdown(f"💊 **Recommended Pesticide:** {pest_info['pesticide']}")
            # Display Pesticide Image
            try:
                # Open the image from the URL
                pesticide_image = Image.open(urllib.request.urlopen(pest_info['image']))
                st.image(pesticide_image, caption=f"{pest_info['pest']} Pesticide", use_column_width=True)
            except Exception as e:
                st.error(f"Error loading pesticide image: {e}.  Please check the image URL.")
        else:
            st.warning("❌ Pest not recognized.")

# ========== Main App ==========
if st.session_state.logged_in:
    pest_detection_page()
else:
    login_page()
