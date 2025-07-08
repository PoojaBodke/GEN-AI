"""
Text-to-Speech (TTS) demo.
We will convert text into audio and save it as a .wav file.
"""

from TTS.api import TTS

# Initialize the model (will download the first time)
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True)

# Text to synthesize
text = "Generative AI is fascinating!"

# Output filename
output_file = "generated_audio.wav"

# Generate and save the audio
tts.tts_to_file(text=text, file_path=output_file)

print(f"Audio saved as {output_file}")
