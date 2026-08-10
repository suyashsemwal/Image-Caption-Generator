# Streamlit web application for generating captions from uploaded images.

import os
import streamlit as st
from PIL import Image

from predict import extract_features, predict_caption


# Page Configuration
st.set_page_config(
    page_title="Image Caption Generator",
    layout="centered"
)


# Application Title

st.title("Image Caption Generator")

st.write("Upload an image")


# Image Upload
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# Generate Caption
if uploaded_file is not None:

    # Display uploaded image
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Generate Caption"):

        with st.spinner("Generating caption..."):

            # Create uploads folder if it doesn't exist
            os.makedirs("uploads", exist_ok=True)

            # Save uploaded image temporarily
            image_path = os.path.join("uploads", uploaded_file.name)

            image.save(image_path)

            # Extract image features
            feature = extract_features(image_path)

            # Generate caption
            caption = predict_caption(feature)

        st.success("Caption Generated Successfully!")

        st.subheader("Generated Caption")

        st.write(caption)