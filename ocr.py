import easyocr as ocr  
import streamlit as st  
from PIL import Image  
import numpy as np  

import cv2  
import google.generativeai as genai

genai.configure(api_key='AIzaSyCbrGlCX60mFMpwmmB_jYJGZs7M21PpTj4')

st.title("Chaduvko bro")

st.markdown("")

# Input selection: Camera or File
input_option = st.radio("Select Input Source:", ["Camera", "File"])

if input_option == "Camera":
    # Camera input
    cap = cv2.VideoCapture(0)

    if st.button("Capture Frame"):
        ret, frame = cap.read()
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

elif input_option == "File":
    # File input
    image = st.file_uploader(label="Upload your image here", type=['png', 'jpg', 'jpeg'])

    if image is not None:
        image = Image.open(image)

# Load OCR model
st.write("Bard response:")
model = genai.GenerativeModel('gemini-pro-vision')
response = model.generate_content(["what is the answer", image])
st.write(response.text)
# Release camera when done
if input_option == "Camera":
    cap.release()
