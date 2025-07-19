import streamlit as st
import uuid 
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os
from faster_whisper import WhisperModel
import sys
import requests

# Add BE folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'BE')))
from STT import wave2vecrecording, wave2vecpath
from text_to_speech import main_tts

# Initialize whisper model once
model = WhisperModel("base", device="cpu", compute_type="int8")


def call_orchestrator(query: str, user_id: str):
    """Call backend FastAPI orchestrator."""
    url = "http://127.0.0.1:8030/query"
    payload = {"query": query, "user_id": user_id}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        return f"API Error: {e}"


def record_audio(duration=5, fs=16000):
    """Record from mic and transcribe using faster-whisper."""
    status_placeholder = st.empty()
    status_placeholder.info(f"Recording for {duration} seconds...")
    text = wave2vecrecording(duration=duration, fs=fs)
    status_placeholder.empty()
    return text


def handle_input(mode):
    """Handle user input based on mode (Text, Record, Upload)."""
    user_input = ""
    audio_bytes = None

    if mode == "Text":
        user_input = st.text_input("Type your message:")

    elif mode == "Record Audio":
        duration = st.slider("Recording duration (sec):", 3, 15, 5)
        if st.button("🎙️ Record Now"):
            user_input = record_audio(duration)

    elif mode == "Upload Audio":
        uploaded = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a"])
        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded.name[-4:]) as tmp:
                tmp.write(uploaded.read())
                tmp.flush()
                user_input = wave2vecpath(tmp.name)
                os.remove(tmp.name)

    return user_input


def display_chat():
    """Render chat history from session state."""
    for entry in st.session_state.chat_history:
        speaker, msg, audio = entry if len(entry) == 3 else (*entry, None)
        st.markdown(f"**{speaker}:** {msg}")
        if audio:
            st.audio(audio, format="audio/wav")


def main():
    st.title("🩺 Clinic Chat Assistant")

    # Input: User ID
    # user_id = st.text_input("Enter your User ID:", key="user_id")
    # if not user_id:
    #     st.warning("Please enter a User ID to continue.")
    #     return
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
        st.info(f"Session ID: {st.session_state.user_id}")
    
    user_id = st.session_state.user_id

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Input mode
    mode = st.radio("Choose input type:", ["Text", "Record Audio", "Upload Audio"])

    # Get user input
    user_input = handle_input(mode)

    if user_input:
        # Call orchestrator
        response = call_orchestrator(user_input, user_id)

        # Convert response to speech if audio mode
        audio_bytes = None
        if mode in ["Record Audio", "Upload Audio"]:
            audio_path = main_tts(response)
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            os.remove(audio_path)

        # Update chat history
        st.session_state.chat_history.append(("You", user_input, audio_bytes))
        st.session_state.chat_history.append(("Bot", response, None))

    # Display the conversation
    display_chat()


if __name__ == "__main__":
    main()
