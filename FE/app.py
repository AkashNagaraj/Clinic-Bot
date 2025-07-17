import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os
from faster_whisper import WhisperModel
import sys
import os

# Add the BE directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'BE')))

from STT import wave2vecrecording, wave2vecpath

# Initialize whisper model
model = WhisperModel("base", device="cpu", compute_type="int8")

# Transcribe audio using faster-whisper
def transcribe_audio(file_path):
    segments, _ = model.transcribe(file_path)
    return " ".join([s.text for s in segments])

# Record mic input
def record_audio(duration=5, fs=16000):
    st.info(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    recorded_text = wave2vecrecording()
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

# Show result
if user_input:
    st.session_state.chat_history.append(("You", user_input))
    # Call the orchestrator & pass all the history API
    # Placeholder response (could be an LLM call)
    response = "Thank you. We'll process your request shortly."
    st.session_state.chat_history.append(("Bot", response))

# Display chat history
for speaker, msg in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {msg}")
