import os
import requests
from git import Repo
from tqdm import tqdm

def download_file(url, filename):
    """
    Helper function to download a file from a given URL
    """
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def setup_environment():
    directories = ["C:/Users/307164/Desktop/avatar_chatbot/sample_data", "C:/Users/307164/Desktop/avatar_chatbot/Wav2Lip"]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
    
    # Clone the Wav2Lip repository if it's not already there
    wav2lip_path = "C:/Users/307164/Desktop/avatar_chatbot/Wav2Lip"
    if not os.listdir(wav2lip_path):  # checks if directory is empty
        Repo.clone_from("https://github.com/Rudrabha/Wav2Lip.git", wav2lip_path)
    
    # Download necessary files
    files = [
        ('https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA', wav2lip_path + '/checkpoints/wav2lip_gan.pth'),
        ('https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth', wav2lip_path + '/face_detection/detection/sfd/s3fd.pth')
    ]
    
    for url, path in files:
        if not os.path.exists(path):
            download_file(url, path)

setup_environment()
