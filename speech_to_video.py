import subprocess

command = 'cd Wav2Lip && python "C:\\Users\\307164\\Desktop\\avatar_chatbot\\Wav2Lip\\inference.py" --checkpoint_path "C:\\Users\\307164\\Desktop\\avatar_chatbot\\Wav2Lip\\checkpoints\\wav2lip_gan.pth" --face "C:\\Users\\307164\\Desktop\\avatar_chatbot\\personna\\audio\\french_translator_animation.mp4" --audio "C:\\Users\\307164\\Desktop\\avatar_chatbot\\personna\\audio\\response.wav"'
subprocess.run(command, shell=True)
