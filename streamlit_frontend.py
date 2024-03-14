import streamlit as st
import time  # For simulating response time
import base64
import threading

# Streamlit page configuration
st.set_page_config(page_title="PDF Contextual Chatbot", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .video-container {
        text-align: center;
        width: 100%; /* Adjust width as needed */
        height: 1000px; /* Adjust height as needed */
        margin: 0 auto 2000px; /* Adjust bottom margin for more space and auto for centering */
    }
    video {
        width: 100%; /* Ensure video fills the container */
        height: auto; /* Maintain aspect ratio */
        object-fit: contain; /* Adjust this to change how the video fits into the container */
        border-radius: 20px;
    }
    .stTextInput>div>div>input {
        color: #4f8bf9;
        border-radius: 20px;
        border: 1px solid #4f8bf9;
        padding: 10px;
        width: 80%; /* Adjust width if needed */
        margin: 0 auto; /* Center the input */
        display: block;
    }
    .chatbox-title {
        margin: 40px 0px 20px 0px; /* Increase spacing around title */
        text-align: center;
    }
    .stMarkdown {
        margin-bottom: 20px; /* Adjust spacing below markdown text */
    }
    </style>
    """, unsafe_allow_html=True)

# UI setup
st.markdown("<h1 style='text-align: center;'>PDF Contextual Chatbot</h1>", unsafe_allow_html=True)

# Function to generate a base64 encoding of the video
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Embedding the video with adjusted styling
def show_video(file_path, height=400):  # You can adjust height as needed
    video_encoded = get_base64_of_bin_file(file_path)
    video_html = f'''
        <div class="video-container" style="max-width: 100%; margin: auto; overflow: hidden;">
            <video loop autoplay muted style="max-width: 100%; height: auto; object-fit: contain; border-radius: 20px;">
                <source src="data:video/mp4;base64,{video_encoded}" type="video/mp4" />
            </video>
        </div>
    '''
    st.components.v1.html(video_html, height=height)

# Display the looped video initially
show_video(r"C:\Users\307164\Desktop\avatar_chatbot\personna\french_translator_animation.mp4")

# Chat interface below the video
st.markdown("<h2 class='chatbox-title'>Your Questions</h2>", unsafe_allow_html=True)
user_input = st.text_input("Type your question here and press Enter:", key="userInput", placeholder="Ask me anything about PDFs...")

def display_response_and_revert_to_looped(user_input):
    with st.spinner('Thinking...'):
        time.sleep(3)  # Simulate response time
        # Display user question and bot response
        st.markdown(f"**You:** {user_input}")
        st.markdown("**PDF Bot:** Here's something interesting about PDFs!")
        
        # Show response video and then revert to looped video
        show_video(r"C:\Users\307164\Desktop\avatar_chatbot\Wav2Lip\results\result_voice.mp4")
        time.sleep(5)  # Duration for the response video, adjust as needed
        show_video(r"C:\Users\307164\Desktop\avatar_chatbot\personna\french_translator_animation.mp4")

if user_input:
    # This function will run the display_response_and_revert_to_looped function in a separate thread
    # This way, it does not block the Streamlit interface while sleeping
    threading.Thread(target=display_response_and_revert_to_looped, args=(user_input,)).start()

# Footer text
st.markdown("### How can I assist you with your PDF files today?", unsafe_allow_html=True)
