from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf

# Set page configuration
st.set_page_config(
    page_title="Malaria Detection App",
    page_icon=":mosquito:",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Add image to top right corner
st.image("img.png", use_column_width=True)

st.title("Malaria Detection App")
st.write("Upload an image of a blood smear to detect the presence of malaria.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Load the pre-trained model
model = tf.keras.models.load_model('malaria_cnn_model.h5')

# Malaria detection function
def detect_malaria(image):
    # Convert image to numpy array
    image_np = np.array(image)
    
    # Resize the image to 128x128
    image_resized = image.resize((128, 128))
    
    # Convert resized image to numpy array
    image_array = np.array(image_resized)
    
    # Preprocess the image (e.g., normalize pixel values)
    image_preprocessed = image_array / 255.0
    image_preprocessed = np.expand_dims(image_preprocessed, axis=0)
    
    # Make a prediction using the model
    predictions = model.predict(image_preprocessed)[0]
    
    # Determine the prediction label and confidence score
    if predictions[0] > 0.5:
        label = 'Uninfected Cell'
        confidence_score = predictions[0]
    else:
        label = 'Parasitized Cell'
        confidence_score = 1 - predictions[0]
    
    return label, confidence_score

# Display the uploaded image and show the prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Create horizontal space to center the image
    col1, col2, col3 = st.columns([1, 2, 1])

    # Resize the image to a smaller size
    smaller_image = image.resize((300, 300))

    # Place the smaller image in the center column
    with col2:
        st.image(smaller_image, caption='Uploaded Cell Image')

    # Run the malaria detection function
    label, confidence_score = detect_malaria(image)

    if label == 'Parasitized Cell':
        st.error("Malaria detected!")
        st.write(f"Confidence score: {confidence_score:.2%}")
    else:
        st.success("No malaria detected!")
        st.write(f"Confidence score: {confidence_score:.2%}")

# Sidebar
st.sidebar.title("About")
st.sidebar.write(
    "This app uses a deep learning model to detect the presence of malaria in blood smear images."
)

st.sidebar.title("How to Use")
st.sidebar.write("1. Upload an image of a blood smear.")
st.sidebar.write("2. The app will analyze the image.")
st.sidebar.write("3. The model predict if it is parasitized or uninfected.")
st.sidebar.write("4. The results will be display below the image alongside the confidence score.")
