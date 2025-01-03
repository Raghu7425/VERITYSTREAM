from dotenv import load_dotenv
load_dotenv()  # Load all environment variables
from youtube_transcript_api import YouTubeTranscriptApi
import streamlit as st
import os
import google.generativeai as genai
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import speech_recognition as sr
import pytube
from pytube import YouTube
import tempfile
import subprocess

# Configure Generative AI with API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load Gemini Pro model
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

# App Configuration
st.set_page_config(page_title="Fact-Checker Application", layout="centered")
st.title("VerityStream - AI-Powered Real-Time Misinformation Detection System")

# Constants
DATA_FILE_PATH = "fake_news_data.xlsx"  # File path for the fake news data

# Function to Load Excel Data
def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        if {'Latitude', 'Longitude', 'Location', 'Fake News', 'Fact'}.issubset(df.columns):
            return df
        else:
            st.error("Data file must include 'Latitude', 'Longitude', 'Location', 'Fake News', and 'Fact' columns.")
            return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Function to Generate Heat Map
def generate_heat_map(data):
    center_lat = data['Latitude'].mean()
    center_lon = data['Longitude'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=5)
    heat_data = data[['Latitude', 'Longitude']].dropna().values.tolist()
    HeatMap(heat_data).add_to(m)

    for _, row in data.iterrows():
        if pd.notnull(row['Latitude']) and pd.notnull(row['Longitude']):
            popup_content = f"""
            <b>Location:</b> {row['Location']}<br>
            <b>Fake News:</b> {row['Fake News']}<br>
            <b>Fact:</b> {row['Fact']}<br>
            <a href="https://twitter.com/intent/tweet?text={row['Fact']}" target="_blank">Share Correct News on Twitter</a>
            """
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=folium.Popup(popup_content, max_width=300)
            ).add_to(m)
    return m

def get_fact_check_response(statement, history):
    """
    Generate a response for fact-checking a statement using Google Generative AI.
    
    Args:
        statement (str): The user's statement to be fact-checked.
        history (list): The conversation history.

    Returns:
        str: AI's response.
    """
    prompt = f"""
    You are an AI fact-checker. Your job is to verify the authenticity and accuracy of the provided statements. 
    Use reliable sources of information to evaluate whether the statement is true, false, or unverifiable. 

    Instructions:
    1. Analyze the statement thoroughly.
    2. Provide a logical and concise evaluation of the statement.
    3. If the statement is false or unverifiable, explain why in simple terms.
    4. Do not include personal opinions or emotions in your response.
    5. Be professional, neutral, and fact-based in your explanation.

    Here is the conversation so far: \n\n
    {''.join([f'{role}: {text} ' for role, text in history])} \n\n
    Now, here is the new statement to fact-check: {statement}\n\n
    Please provide your analysis."""

    response = chat.send_message(prompt, stream=False)
    return response



# Function to Convert Video to Audio
def convert_video_to_audio(video_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        audio_path = temp_audio.name

    command = [
        "ffmpeg",
        "-i", video_file,
        "-vn",
        "-acodec", "mp3",
        audio_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path

# Function to fetch YouTube live transcript

from youtube_transcript_api import YouTubeTranscriptApi
import re

def fetch_youtube_live_transcript(url, fact_check_function):
    """
    Fetch the YouTube live transcript and perform fact-checking for each statement.

    Args:
        url (str): The YouTube video URL.
        fact_check_function (callable): Function to perform fact-checking on a statement.

    Returns:
        str: Transcript with fact-check results or an error message.
    """
    try:
        # Extract the video ID from the URL using regex
        video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
        if not video_id_match:
            return "Error: Invalid YouTube URL. Could not extract video ID."
        
        video_id = video_id_match.group(1)

        # Fetch the transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        
        # Perform fact-checking for each statement
        fact_check_results = []
        for entry in transcript:
            statement = entry['text']
            fact_check_response = fact_check_function(statement, history=[])  # Adjust history as needed
            fact_check_results.append(f"Statement: {statement}\nFact-Check: {fact_check_response}\n")
        
        return "\n".join(fact_check_results)

    except Exception as e:
        return f"Error fetching transcript: {str(e)}"


# Real-time Speech to Text with Fact-Checking
def real_time_speech_to_text_with_fact_check():
    """
    Capture real-time speech, convert it to text, and perform fact-checking.
    """
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    try:
        with mic as source:
            st.write("Listening... Speak now.")
            audio = recognizer.listen(source)
            speech_text = recognizer.recognize_google(audio)
            st.write(f"Recognized Speech: {speech_text}")
            
            # Perform fact-checking on the recognized speech
            if speech_text:
                history = []  # Update with previous history if needed
                response = get_fact_check_response(speech_text, history)
                response_filter = response.candidates[0].content.parts
                response_text = ' '.join(part.text for part in response_filter)
                st.subheader("Fact-Check Result")
                st.write(response_text)
            else:
                st.warning("No speech detected. Please try again.")

    except sr.UnknownValueError:
        st.error("Could not understand the speech. Please try again.")
    except sr.RequestError:
        st.error("Error with the speech recognition service.")


# Load Excel Data
data = load_data(DATA_FILE_PATH)

# Sidebar Navigation
menu = st.sidebar.radio("Select Feature", [
    "Text Fact-Check",
    "Audio/Video Fact-Check",
    "YouTube Live Fact-Check",
    "Real-Time Speech Fact-Check",
    "Fake News Data",
    "Fake News Heat Map"
])

if menu == "Text Fact-Check":
    st.subheader("Text Fact-Check")
    statement = st.text_area("Enter a statement to fact-check")
    if st.button("Check Fact"):
        if statement:
            response = get_fact_check_response(statement, chat.history)
            # Add user query and response to session state chat history
            response_filter = response.candidates[0].content.parts
            response_text = ' '.join(part.text for part in response_filter)
            st.write(response_text)
        else:
            st.error("Please enter a statement.")

elif menu == "Audio/Video Fact-Check":
    st.subheader("Audio/Video Fact-Check")
    uploaded_file = st.file_uploader("Upload an audio or video file", type=["mp4", "mp3"])
    if uploaded_file and st.button("Process File"):
        audio_path = convert_video_to_audio(uploaded_file.name)
        st.write(f"Audio extracted to: {audio_path}")

elif menu == "YouTube Live Fact-Check":
    st.subheader("YouTube Live Fact-Check")
    url = st.text_input("Enter YouTube Video URL")
    if st.button("Fetch and Fact-Check Transcript"):
        if url:
            with st.spinner("Processing..."):
                transcript_results = fetch_youtube_live_transcript(url, get_fact_check_response)
                st.text_area("Transcript with Fact-Check Results", value=transcript_results, height=400)
        else:
            st.error("Please enter a valid YouTube video URL.")


elif menu == "Real-Time Speech Fact-Check":
    st.subheader("Real-Time Speech Fact-Check")
    st.write("Click the button below and start speaking.")
    
    if st.button("Start Listening"):
        real_time_speech_to_text_with_fact_check()

elif menu == "Fake News Data":
    if data is not None:
        st.subheader("Fake News Data")
        st.write("Browse the list of fake news and their corresponding facts.")
        st.dataframe(data)
    else:
        st.error("Failed to load the data file. Ensure the file exists and is properly formatted.")

elif menu == "Fake News Heat Map":
    if data is not None:
        st.subheader("Fake News Heat Map")
        st.write("Visualizing the spread of fake news across locations.")
        heat_map = generate_heat_map(data)
        st_folium(heat_map, width=800, height=500)
    else:
        st.error("Failed to load the data file. Ensure the file exists and is properly formatted.")
