from datasets import load_dataset
import torch
from transformers import pipeline
import soundfile as sf
import subprocess

def get_speaker_embedding(embeddings_dataset, index):
    return torch.tensor(embeddings_dataset[index]["xvector"]).unsqueeze(0)

def speech_generation(response, fn):
    embeddings_dataset = load_dataset("cmu-arctic-xvectors", split="validation")
    audio_embedding = get_speaker_embedding(embeddings_dataset, 7306)
    synthesizer = pipeline("text-to-speech", model="microsoft/speecht5_tts")
    result = synthesizer(response, forward_params={"speaker_embeddings": audio_embedding})
    sf.write(rf"C:\\Users\\307164\\Desktop\\avatar_chatbot\\personna\\audio\\{fn}.wav", result["audio"], samplerate=22050)
    command = f'cd Wav2Lip && python "C:\\Users\\307164\\Desktop\\avatar_chatbot\\Wav2Lip\\inference.py" --checkpoint_path "C:\\Users\\307164\\Desktop\\avatar_chatbot\\Wav2Lip\\checkpoints\\wav2lip_gan.pth" --face "C:\\Users\\307164\\Desktop\\avatar_chatbot\\personna\\audio\\french_translator_animation.mp4" --audio "C:\\Users\\307164\\Desktop\\avatar_chatbot\\personna\\audio\\{fn}.wav"'
    subprocess.run(command, shell=True)

introduction_generation = speech_generation("Hi, I am Alphadata ChatBot. Nice to meet you. What question do you want me to answer about the uploaded pdf document?", "intro")
#filler_generation = speech_generation("What else would you like to know?", "transition")
