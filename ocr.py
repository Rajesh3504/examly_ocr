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
st.write("Answer:")
model = genai.GenerativeModel('gemini-pro-vision')
ans="what is the answer"
response = model.generate_content([ans, image])
st.write(response.text)

prompt = st.text_input("Enter your prompt:")

if st.button("Generate Response"):
        
        response1 = model.generate_content([prompt, image])

st.write(response1.text)

# Release camera when done
if input_option == "Camera":
    cap.release()
