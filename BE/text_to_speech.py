from TTS.api import TTS

def main():
    # Init TTS
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

    # Run TTS
    tts.tts_to_file(text="Thank you for visiting our clinic today. Your appointment has been successfully recorded, and the doctor will see you shortly. Please ensure you have your medical records ready for consultation. If you have any questions regarding your prescription, follow-up visits, or insurance coverage, feel free to speak with the front desk or our attending nurse.", file_path="../data/output.wav")

if __name__=="__main__":
    main()