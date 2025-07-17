from RealtimeSTT import AudioToTextRecorder
from faster_whisper import WhisperModel
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os
from faster_whisper import WhisperModel


def wave2vecpath(audio_path="../data/book_slot.m4a"):
    model_size = "base"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, beam_size=5)

    text_data = []
    for segment in segments:
        text_data.append(segment.text)

    return " ".join(text_data)


def wave2vecrecording(duration=5, fs=16000):
    # Record audio from mic
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("Recording finished.")

    # Save audio to temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        write(tmp_file.name, fs, audio)
        audio_path = tmp_file.name

    # Transcribe with faster-whisper
    model_size = "base"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, beam_size=5)

    # Collect text
    text_data = []
    for segment in segments:
        text_data.append(segment.text)

    # Cleanup temp file
    os.remove(audio_path)

    return " ".join(text_data)


def main():
    wave2vec()

if __name__=="__main__":
    main()

