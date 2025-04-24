import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json
import os
from huggingface_hub import hf_hub_download

# ========== Constants ==========
USER_FILE = "users.json"
IMG_SIZE = (380, 380)

# ========== Class Labels ==========
class_labels = {
    'aphids': {'id': 0, 'pesticide': 'Imidacloprid', 'image': 'assets/pesticide_images/aphid_pesticide.jpg'},
    'armyworm': {'id': 1, 'pesticide': 'Lambda-cyhalothrin', 'image': 'assets/pesticide_images/armyworm_pesticide.jpg'},
    'beetle': {'id': 2, 'pesticide': 'Carbaryl', 'image': 'assets/pesticide_images/beetle_pesticide.jpg'},
    'bollworm': {'id': 3, 'pesticide': 'Chlorpyrifos', 'image': 'assets/pesticide_images/bollworm_pesticide.jpg'},
    'grasshopper': {'id': 4, 'pesticide': 'Malathion', 'image': 'assets/pesticide_images/grasshopper_pesticide.jpg'},
    'mites': {'id': 5, 'pesticide': 'Abamectin', 'image': 'assets/pesticide_images/mites_pesticide.jpg'},
    'mosquito': {'id': 6, 'pesticide': 'Temephos', 'image': 'assets/pesticide_images/mosquito_pesticide.jpg'},
    'sawfly': {'id': 7, 'pesticide': 'Spinosad', 'image': 'assets/pesticide_images/sawfly_pesticide.jpg'},
    'stem_borer': {'id': 8, 'pesticide': 'Quinalphos', 'image': 'assets/pesticide_images/stem_borer_pesticide.jpg'}
}

# Reverse mapping from ID to label name
index_to_label = {v['id']: k for k, v in class_labels.items()}

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
    model_path = hf_hub_download(repo_id="hiddu2004/hello", filename="final_model(2).h5")
    return load_model(model_path)

model = load_pest_model()

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
        predicted_label = index_to_label.get(class_idx)
        pest_info = class_labels.get(predicted_label)

        if pest_info:
            st.success(f"🪲 **Detected Pest:** {predicted_label.replace('_', ' ').title()}")
            st.markdown(f"💊 **Recommended Pesticide:** {pest_info['pesticide']}")
            st.image(pest_info['image'], caption=f"{predicted_label.replace('_', ' ').title()} Pesticide", use_column_width=True)
        else:
            st.warning("❌ Pest not recognized.")

# ========== Main App ==========
if st.session_state.logged_in:
    pest_detection_page()
else:
    login_page()
