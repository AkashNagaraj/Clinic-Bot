import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os
from faster_whisper import WhisperModel
import sys
import requests

# Add the BE directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'BE')))

from STT import wave2vecrecording, wave2vecpath
from text_to_speech import main_tts

# Initialize whisper model
model = WhisperModel("base", device="cpu", compute_type="int8")

# Transcribe audio using faster-whisper
def transcribe_audio(file_path):
    segments, _ = model.transcribe(file_path)
    return " ".join([s.text for s in segments])

# Record mic input
def record_audio(duration=5, fs=16000):
    status_placeholder = st.empty()
    status_placeholder.info(f"Recording for {duration} seconds...")

    # Perform recording + transcription
    recorded_text = wave2vecrecording(duration=duration, fs=fs)
    status_placeholder.empty()

    # st.write(recorded_text)
    return recorded_text

# Streamlit UI
st.title("🩺 Clinic Chat Assistant")

# Store chat in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input mode
mode = st.radio("Choose input type:", ["Text", "Record Audio", "Upload Audio"])

# Input section
user_input = ""
if mode == "Text":
    user_input = st.text_input("Type your message:")
elif mode == "Record Audio":
    duration = st.slider("Recording duration (sec):", 3, 15, 5)
    if st.button("🎙️ Record Now"):
        user_input = record_audio(duration)
elif mode == "Upload Audio":
    uploaded = st.file_uploader("Upload audio file (.wav/.mp3/.m4a)", type=["wav", "mp3", ".m4a"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded.name[-4:]) as tmp:
            tmp.write(uploaded.read())
            tmp.flush()
            user_input = wave2vecpath(tmp.name) #transcribe_audio(tmp.name)
            os.remove(tmp.name)

def call_orchestrator(query: str):
    url = "http://127.0.0.1:8030/query"
    payload = {"query": query}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"Error: {response.status_code} - {response.text}"

# Show result
if user_input:
    response = call_orchestrator(user_input)

    # If it's an audio input, store both text and audio
    if mode in ["Record Audio", "Upload Audio"]:
        # Save audio to a temporary file or bytes
        audio_bytes = None
        audio_path = main_tts(response)
        audio_bytes = open(audio_path, "rb").read()
        os.remove(audio_path)
        # Add user message with audio
        st.session_state.chat_history.append(("You", user_input, audio_bytes))
    else:
        # Just text
        st.session_state.chat_history.append(("You", user_input, None))

    st.session_state.chat_history.append(("Bot", response, None))

# Display chat history
for entry in st.session_state.chat_history:
    if len(entry) == 3:
        speaker, msg, audio = entry
    else:
        speaker, msg = entry
        audio = None

    st.markdown(f"**{speaker}:** {msg}")
    if audio:
        st.audio(audio, format='audio/wav')

