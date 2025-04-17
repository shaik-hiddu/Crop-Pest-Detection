import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your model
model = load_model("model/my_model.h5")  # Adjust path if necessary
img_size = (224, 224)

# Class labels and pesticide information (for 9 classes)
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

st.title("🌿 Crop Pest Detector")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image_data = Image.open(uploaded_file).convert('RGB')
    st.image(image_data, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = image_data.resize(img_size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)
    class_idx = np.argmax(prediction, axis=1)[0]
    
    # Get pest and pesticide info
