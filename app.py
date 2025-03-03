import requests
import streamlit as st
from PIL import Image
from streamlit_lottie import st_lottie
import os

st.set_page_config(page_title="G-LEARN", page_icon="📚", layout="wide")

# Load Lottie Animation
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding = load_lottieurl("https://lottie.host/7f698e81-6288-428b-8ee9-2b1930b763a3/Jg71B322ri.json")

# Load Images
img_contact_form = Image.open("yt_contact_form.png") if os.path.exists("yt_contact_form.png") else None
img_lottie_animation = Image.open("yt_lottie_animation.png") if os.path.exists("yt_lottie_animation.png") else None

st.title("Welcome to G-LEARN!")

with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)

    with left_column:
        st.header("Continue as")
        st.write("##")
        user_type = st.radio("Select your role:", ["Parent/Teacher", "Student"])

        picture = st.camera_input("Take a picture")
        if picture:
            st.image(picture, caption="Your Picture")

        audio_input = st.audio_input("Record a voice message")
    with right_column:
        if lottie_coding:
            st_lottie(lottie_coding, height=300, key="coding")

with st.container():
    st.write("---")
    st.header("My Projects")
    st.write("##")
    image_column, text_column = st.columns((1, 2))

    with image_column:
        if img_lottie_animation:
            st.image(img_lottie_animation, caption="Project Preview")

    with text_column:
        st.write("Discover our latest projects and innovations in e-learning.")
        st.markdown("[Check it out!](#)")

with st.container():
    st.write("---")
    st.header("Get in Touch with Us")
    st.write("##")
    contact_form = """
    <form action="https://formsubmit.co/snehashirshad09@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your Name" required>
        <input type="email" name="email" placeholder="Your Email" required>
        <textarea name="message" placeholder="Your Message Here" required></textarea> 
        <button type="submit">Send</button>
    </form>
    """ 

    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)

    with right_column:
        if os.path.exists("web.image"):
            st.image("web.image", caption="Contact Us", use_column_width=True)
        else:
            st.error("Image 'web.image' not found!")

