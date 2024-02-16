import easyocr as ocr  
import streamlit as st  
from PIL import Image  
import numpy as np  

import cv2  
import google.generativeai as genai

genai.configure(api_key='AIzaSyACbzAawVBFyGxHTpl8NCiceYPiKhcdvUw')

st.title("Chaduvko bro")

st.markdown("")

#    Input selection: Camera or File
input_option = st.radio("Select Input Source:", ["Camera", "File"])

if input_option == "Camera":
    # Camera input
    cap = cv2.VideoCapture(0)

    if st.button("Capture Frame"):
        ret, frame = cap.read()
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

elif input_option == "File":
    # File input
    image=''
    image = st.file_uploader(label="Upload your image here", type=['png', 'jpg', 'jpeg'])

    if image is not None:
        image = Image.open(image)

st.spinner("Ruko zara sabar karo")
st.image(image)

st.write("Answer:")
model = genai.GenerativeModel('gemini-pro-vision')
response = 'answer'
answer="what is the answer"
response = model.generate_content([answer, image])
st.write(response.text)

prompt = st.text_input("Enter your prompt:")
response1 = 'answer'
if st.button("Generate Response"):
        response1= model.generate_content([prompt, image])
        st.write(response1.text)
# Release camera when done
if input_option == "Camera":
    cap.release()
