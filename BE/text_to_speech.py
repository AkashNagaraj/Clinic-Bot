from TTS.api import TTS

def main_tts(text=""):
    # Init TTS
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

    # Run TTS
    tmp_file_path = "../data/output.wav"
    print('Text from TTS : ', text)
    tts.tts_to_file(text=text, file_path = tmp_file_path)
    return tmp_file_path


if __name__=="__main__":
    _ = main_tts("Hi how are you")