# # streamlit_app.py

# import streamlit as st
# from ultralytics import YOLO
# import cv2
# import numpy as np
# from PIL import Image

# # Load the model
# model = YOLO("C:\\Users\\gargs\\OneDrive\\Desktop\\traffic sign\\Trafic sign detection\\runs\\classify\\train4\\weights\\best.pt")  # Update this with your model path

# # Set up the Streamlit app UI
# st.title("Traffic Sign Detection")
# st.write("Upload an image to detect traffic signs!")

# # Image upload section
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Load and display the uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)
#     st.write("Processing...")

#     # Convert the uploaded image to the format YOLO expects
#     image_np = np.array(image)
#     results = model.predict(image)
#     pred = results[0].names[results[0].probs.top1]
#     st.write("Prediction : ", pred)
# else:
#     st.write("Please upload an image to get started.")


# streamlit_app.py

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load the model
model = YOLO("C:\\Users\\gargs\\OneDrive\\Desktop\\traffic sign\\Trafic sign detection\\runs\\classify\\train4\\weights\\best.pt")  # Update this with your model path

# Set up the Streamlit app UI
st.title("Traffic Sign Recognition")
st.write("Upload an image or select a sample image to detect traffic signs!")

# Sample images list (Add paths to your sample images)
sample_images = {
    "Sample Image 1": "C:\\Users\\gargs\\OneDrive\\Desktop\\traffic sign\\Trafic sign detection\\test\\Go Right\\024_1_0026.png",
    "Sample Image 2": "C:\\Users\\gargs\\OneDrive\\Desktop\\traffic sign\\Trafic sign detection\\test\\Dont Go Left\\011_1_0014.png",
    "Sample Image 3": "C:\\Users\\gargs\\OneDrive\\Desktop\\traffic sign\\Trafic sign detection\\test\\Dont Go straight or Right\\009_1_0001.png",
    "Sample Image 4": "C:\\Users\\gargs\\OneDrive\\Desktop\traffic sign\\Trafic sign detection\\test\\Dont Go straight or Right\\009_1_0001.png",
    "Sample Image 5": "C:\\Users\\gargs\\OneDrive\\Desktop\traffic sign\\Trafic sign detection\\test\\Go right or straight\\043_1_0036.png",
    "Sample Image 6": "C:\\Users\\gargs\\OneDrive\\Desktop\\traffic sign\\Trafic sign detection\\test\\Under Construction\\048_1_0003.png",
    "Sample Image 7": "C:\\Users\\gargs\\OneDrive\\Desktop\\traffic sign\\Trafic sign detection\\test\\Zebra Crossing\\035_1_0014.png",
    "Sample Image 8": "C:\\Users\\gargs\\OneDrive\\Desktop\\traffic sign\\Trafic sign detection\\test\\Zebra Crossing\\035_1_0010.png",
    "Sample Image 9": "C:\\Users\\gargs\\OneDrive\\Desktop\\traffic sign\\Trafic sign detection\\test\\Dont overtake from Left\\014_1_0013.png",
    "Sample Image 10": "C:\\Users\\gargs\\OneDrive\\Desktop\\traffic sign\\Trafic sign detection\\test\\Dangerous curve to the left\\038_1_0012.png",
    "Sample Image 11": "C:\\Users\\gargs\\OneDrive\\Desktop\\traffic sign\\Trafic sign detection\\test\\Go Left\\022_0005.png",
    "Sample Image 12": "C:\\Users\\gargs\\OneDrive\\Desktop\\traffic sign\\Trafic sign detection\\test\\Dangerous curve to the right\\039_1_0013.png",
    "Sample Image 13": "C:\\Users\\gargs\\OneDrive\\Desktop\\traffic sign\\Trafic sign detection\\test\\Danger Ahead\\050_1_0013.png",
    "Sample Image 14": "C:\\Users\\gargs\OneDrive\\Desktop\\traffic sign\\Trafic sign detection\\test\\No entry\\055_1_0014.png",
    "Sample Image 15": "C:\\Users\\gargs\\OneDrive\\Desktop\traffic sign\\Trafic sign detection\\test\\No Car\\016_1_0015.png"
}

# Dropdown for selecting sample images
selected_sample = st.selectbox("Choose a sample image or upload your own:", ["None"] + list(sample_images.keys()))

# Image upload section
uploaded_file = st.file_uploader("Or upload an image...", type=["jpg", "jpeg", "png"])

# Check if a sample image is selected
if selected_sample != "None":
    # Load the selected sample image
    image_path = sample_images[selected_sample]
    image = Image.open(image_path)
    st.image(image, caption=f"Selected Sample Image: {selected_sample}", use_column_width=True)
    st.write("Processing...")
    
elif uploaded_file is not None:
    # Load and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")
    
else:
    st.write("Please select a sample image or upload an image to get started.")
    image = None

# Run prediction if an image is selected or uploaded
if image is not None:
    # Convert the image to the format YOLO expects
    image_np = np.array(image)
    results = model.predict(image)
    pred = results[0].names[results[0].probs.top1]
    st.write("Prediction:", pred)

