from dotenv import load_dotenv
load_dotenv()

import openai
import streamlit as st
from moviepy.editor import VideoFileClip
import os
import io
from src.utils import upload_to_s3, generate_presigned_url
from openai import OpenAI


BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
client = OpenAI()


# Function to detect names using GPT-4 Turbo
def detect_names(text):
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that identifies names in Hindi text."},
            {"role": "user", "content": f"Identify all names in this Hindi text. Output only the names, separated by commas: {text}"}
        ]
    )
    return response.choices[0].message.content.strip().split(', ')


# Function to transcribe audio using Whisper
def transcribe_audio(audio_file_path):
    with open(audio_file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
            )
    return transcription.text


# Function to replace detected names in text with user-specified names
def replace_names_in_text(text, name_mapping):
    for original_name, custom_name in name_mapping.items():
        text = text.replace(original_name, custom_name)
    return text


# Function to extract audio from video
def extract_audio(video_file):
    clip = VideoFileClip(video_file)
    audio = clip.audio
    audio_file = "extracted_audio.mp3"
    audio.write_audiofile(audio_file)
    return audio_file



# Streamlit app
st.title("Video Dub | Clone")

# Use session state to store persistent data
if 'processed' not in st.session_state:
    st.session_state.processed = False
    st.session_state.transcription = ""
    st.session_state.detected_names = []

# File uploader
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_video is not None and not st.session_state.processed:
    # Process the video
    with st.spinner("Processing video..."):
        # Save uploaded video temporarily
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())

        # Extract audio from video
        audio_path = extract_audio("temp_video.mp4")

        # Transcribe the audio
        st.session_state.transcription = transcribe_audio(audio_path)

        # Detect names in the transcription
        st.session_state.detected_names = detect_names(st.session_state.transcription)

        # Clean up temporary files
        os.remove("temp_video.mp4")
        os.remove(audio_path)

        st.session_state.processed = True

if st.session_state.processed:
    # Display the original transcription
    st.subheader("Original Transcription")
    st.write(st.session_state.transcription)

    # Display detected names
    st.subheader("Detected Names")
    st.write(", ".join(st.session_state.detected_names))

    # Get name replacements from user input
    name_mapping = {}
    for name in st.session_state.detected_names:
        custom_name = st.text_input(f"Replace '{name}' with (leave blank to skip):", key=f"input_{name}")
        if custom_name:
            name_mapping[name] = custom_name

    # Apply name replacements
    if st.button("Apply Name Replacements"):
        modified_transcription = replace_names_in_text(st.session_state.transcription, name_mapping)

        # Display the modified transcription
        st.subheader("Modified Transcription")
        st.write(modified_transcription)
        