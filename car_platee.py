import easyocr 
import streamlit as st
import torch
import cv2
from PIL import Image
import io
import numpy as np
import pandas as pd 
from numba import jit,njit,vectorize,cuda,uint32,f8,uint8


@jit 
def cordinates_detection(img, reader): 
    # Convert raw image bytes to a format that OpenCV can handle
    image = Image.open(io.BytesIO(img))
    image = np.array(image)
    
    if image.shape[2] == 4:  # Check if the image has an alpha channel (indicating PNG)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Ensure image is in CHW format
    image = image.transpose((2, 0, 1))

    # Apply YOLO detection
    predict_image = yolo(image)
    current_coordinates = predict_image.xyxy[0][:, :-1]
    current_coordinates_cpu = current_coordinates.cpu().numpy()

    # Assuming results is a list of coordinates for one photo
    if len(current_coordinates_cpu) > 0:  # Check if the list is not empty
        row = current_coordinates_cpu[0]  # Take the first and only row
        confidence_index = -1  # Adjust this index based on your actual data structure

        if row[confidence_index] >= 0.5:  # Take img with 0.5 confidence
            xmin, ymin, xmax, ymax = row[:4]
            plate = image[:, int(ymin):int(ymax), int(xmin):int(xmax)].transpose((1, 2, 0))

            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            blurred = cv2.bilateralFilter(gray, 17, 15, 15)
            text = reader.readtext(blurred)
            text = ' '.join([t[1] for t in text])

            # Convert plate back to RGB format for displaying with Streamlit
            plate_rgb = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)

            plot_img = image.transpose((1, 2, 0)).copy()

            cv2.rectangle(plot_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)  # BBox
            cv2.rectangle(plot_img, (int(xmin), int(ymin - 20)), (int(xmax), int(ymin)), (0, 255, 0), -1)  # Text label background
            final_img = cv2.putText(plot_img, f"{text}", (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Convert final_img back to RGB format for displaying with Streamlit
            final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

            return final_img_rgb, text, plate_rgb  
    else:
        return None, None, None  # No plate detected

# Load the YOLO model
yolo = torch.hub.load('ultralytics/yolov5', 'custom', path='b.pt', force_reload=True)



# Create an OCR reader
reader = easyocr.Reader(['en'])

# Create the Streamlit app
st.title('Car Plate Detection and Recognition')
st.write('This app uses YOLO and EasyOCR to detect and recognize car plates from images.')

# Upload an image file
uploaded_file = st.file_uploader('Choose an image file', type=['jpg', 'png', 'PNG', 'jpeg'])
st.markdown(
        """
           <style>
.main {
    background-image: url("https://images.unsplash.com/photo-1603386329225-868f9b1ee6c9?q=80&w=2069&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
}
</style>,

        """, unsafe_allow_html=True
    )
# Selectbox for applying effects
selected_option = st.selectbox('Choose an option:', ['Original', 'Apply Blur', 'Apply Sharpen', 'Read Plate'])

# Process and display the image
if uploaded_file is not None:
    # Read the image
    img = uploaded_file.read()

    # Apply the YOLO and EasyOCR functions
    annotated_img, text, plate_img = cordinates_detection(img, reader)

    # Apply selected option
    if selected_option == 'Apply Blur':
        blured_img = cv2.GaussianBlur(annotated_img, (15, 15), 0)
        st.image(blured_img, caption='Bluring car ')
    elif selected_option == 'Apply Sharpen':
        sharped_img = cv2.filter2D(annotated_img, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
        st.image(sharped_img, caption='Bluring car ')
    elif selected_option == 'Read Plate':
        st.write('The car plate number is:', text)
        st.image(annotated_img, caption='Detected car plate')
        st.image(plate_img, caption='Car plate')

    