import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json
import os

# ========== Constants ==========
USER_FILE = "users.json"
MODEL_PATH = "final_model.h5"  # Replace with actual path if loading locally
IMG_SIZE = (380, 380)

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
    model_path = hf_hub_download(repo_id="hiddu2004/hello", filename="final_model.h5")
    return load_model(model_path)

model = load_pest_model()

# ========== Class Labels ==========
class_labels = {
    0: {'pest': 'Aphid', 'pesticide': 'Pyrethroids', 'pesticide_img': 'assets/pesticide_images/aphid_pesticide.jpg'},
    1: {'pest': 'Armyworm', 'pesticide': 'Bacillus thuringiensis', 'pesticide_img': 'assets/pesticide_images/armyworm_pesticide.jpg'},
    2: {'pest': 'Caterpillar', 'pesticide': 'Insecticidal Soap', 'pesticide_img': 'assets/pesticide_images/caterpillar_pesticide.jpg'},
    3: {'pest': 'Whitefly', 'pesticide': 'Neem Oil', 'pesticide_img': 'assets/pesticide_images/whitefly_pesticide.jpg'},
    4: {'pest': 'Thrips', 'pesticide': 'Spinosad', 'pesticide_img': 'assets/pesticide_images/thrips_pesticide.jpg'},
    5: {'pest': 'Leafhopper', 'pesticide': 'Malathion', 'pesticide_img': 'assets/pesticide_images/leafhopper_pesticide.jpg'},
    6: {'pest': 'Root Knot Nematode', 'pesticide': 'Fumigants', 'pesticide_img': 'assets/pesticide_images/root_knot_nematode_pesticide.jpg'},
    7: {'pest': 'Cucumber Beetle', 'pesticide': 'Diazinon', 'pesticide_img': 'assets/pesticide_images/cucumber_beetle_pesticide.jpg'},
    8: {'pest': 'Aphid (Green)', 'pesticide': 'Chlorpyrifos', 'pesticide_img': 'assets/pesticide_images/green_aphid_pesticide.jpg'},
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

        # Preprocessing
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
            st.image(pest_info['pesticide_img'], caption=f"{pest_info['pest']} Pesticide", use_column_width=True)
        else:
            st.warning("❌ Pest not recognized.")

# ========== Main App ==========
if st.session_state.logged_in:
    pest_detection_page()
else:
    login_page()

