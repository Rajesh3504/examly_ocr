import easyocr as ocr  
import streamlit as st  
from PIL import Image  
import numpy as np  
import cv2  
import google.generativeai as genai

genai.configure(api_key='AIzaSyCbrGlCX60mFMpwmmB_jYJGZs7M21PpTj4')

# title
st.title("Answers kavala bro")



st.markdown("")

# Input selection: Camera or File
input_option = st.radio("Select Input Source:", ["Camera", "File"])

if input_option == "Camera":
    # Camera input
    cap = cv2.VideoCapture(0)

    if st.button("Capture Frame"):
        ret, frame = cap.read()
        input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

elif input_option == "File":
    # File input
    image = st.file_uploader(label="Upload your image here", type=['png', 'jpg', 'jpeg'])

    if image is not None:
        input_image = Image.open(image)

# Load OCR model
@st.cache_resource
def load_model():
    reader = ocr.Reader(['en'], model_storage_directory='.')
    return reader

reader = load_model()  # Load model
model = genai.GenerativeModel('gemini-pro-vision')
if 'input_image' in locals():
    st.image(input_image)  # Display image

    with st.spinner("Please wait, processing..."):

        result = reader.readtext(np.array(input_image))

        result_text = []  # Empty list for results

        for text in result:
            result_text.append(text[1])

        st.write("Extracted Text:")
        st.write(result_text)

        # Join the extracted text for ChatGPT query
        query = " ".join(result_text)

        # Use ChatGPT to get a response
        if query:
             model = genai.GenerativeModel('gemini-pro')
             response = model.generate_content(query)
st.write(response.text)
if input_option == "Camera":
    cap.release()
