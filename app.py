import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json
import os
from huggingface_hub import hf_hub_download
import urllib.request

# ========== Constants ========== 
USER_FILE = "users.json"
IMG_SIZE = (380, 380)

# ========== User Authentication ========== 
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

# ========== Load Model from Hugging Face ========== 
@st.cache_resource
def load_pest_model():
    model_path = hf_hub_download(repo_id="hiddu2004/hello", filename="final_model2.h5")
    return load_model(model_path)

model = load_pest_model()

# ========== Class Labels with Image URLs ========== 
class_labels = {
    0: {
        'pest': 'aphid',
        'pesticide': 'Imidacloprid',
        'image': 'https://huggingface.co/hiddu2004/hello/resolve/main/pesticide_images/aphid_pesticide.jpg'
    },
    1: {
        'pest': 'armyworm',
        'pesticide': 'Lambda-cyhalothrin',
        'image': 'https://huggingface.co/hiddu2004/hello/resolve/main/pesticide_images/armyworm_pesticide.jpg'
    },
    2: {
        'pest': 'beetle',
        'pesticide': 'Carbaryl',
        'image': 'https://huggingface.co/hiddu2004/hello/resolve/main/pesticide_images/beetle_pesticide.jpeg'
    },
    3: {
        'pest': 'bollworm',
        'pesticide': 'Chlorpyrifos',
        'image': 'https://huggingface.co/hiddu2004/hello/resolve/main/pesticide_images/bollworm_pesticide.jpeg'
    },
    4: {
        'pest': 'grasshopper',
        'pesticide': 'Malathion',
        'image': 'https://huggingface.co/hiddu2004/hello/resolve/main/pesticide_images/grasshopper_pesticide.jpeg'
    },
    5: {
        'pest': 'mites',
        'pesticide': 'Abamectin',
        'image': 'https://huggingface.co/hiddu2004/hello/resolve/main/pesticide_images/mites_pesticide.jpg'
    },
    6: {
        'pest': 'mosquito',
        'pesticide': 'Temephos',
        'image': 'https://huggingface.co/hiddu2004/hello/resolve/main/pesticide_images/mosquito_pesticide.jpeg'
    },
    7: {
        'pest': 'sawfly',
        'pesticide': 'Spinosad',
        'image': 'https://huggingface.co/hiddu2004/hello/resolve/main/pesticide_images/sawfly_pesticide.jpeg'
    },
    8: {
        'pest': 'stem_borer',
        'pesticide': 'Quinalphos',
        'image': 'https://huggingface.co/hiddu2004/hello/resolve/main/pesticide_images/stem_borer_pesticide.jpg'
    }
}


# ========== Session State ========== 
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ========== Login/Register Interface ========== 
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
                st.error("Invalid username or password.")

    with tab2:
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        if st.button("Register"):
            if save_user(new_user, new_pass):
                st.success("User registered! You can now log in.")
            else:
                st.warning("Username already exists.")

# ========== Pest Detection Interface ========== 
def pest_detection_page():
    st.title("🌿 Crop Pest Detection")
    st.write(f"👤 Logged in as **{st.session_state.username}**")

    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

    uploaded_file = st.file_uploader("📷 Upload pest image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image_data = Image.open(uploaded_file).convert("RGB")
        st.image(image_data, caption="Uploaded Image", use_column_width=True)

        # Preprocess and Predict
        img = image_data.resize(IMG_SIZE)
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        class_idx = np.argmax(prediction, axis=1)[0]
        st.write(f"Prediction: {prediction}, Class Index: {class_idx}")  # Debugging output
        pest_info = class_labels.get(class_idx)

        if pest_info:
            st.success(f"🪲 **Detected Pest:** {pest_info['pest']}")
            st.markdown(f"💊 **Recommended Pesticide:** {pest_info['pesticide']}")
            try:
                pesticide_img = Image.open(urllib.request.urlopen(pest_info['image']))
                st.image(pesticide_img, caption=f"{pest_info['pest']} Pesticide", use_column_width=True)
            except Exception as e:
                st.error(f"Couldn't load pesticide image: {e}")
        else:
            st.warning("❌ Pest not recognized.")

# ========== Run App ========== 
if st.session_state.logged_in:
    pest_detection_page()
else:
    login_page()
