!git clone https://github.com/zabique/Wav2Lip

!wget 'https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA' -O 'C:\Users\307164\Desktop\avatar_chatbot\Wav2Lip\checkpoints\wav2lip_gan.pth'
(or)
Invoke-WebRequest -Uri "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA" -OutFile "C:\Users\307164\Desktop\avatar_chatbot\Wav2Lip\checkpoints\wav2lip_gan.pth"

!pip install https://raw.githubusercontent.com/AwaleSajil/ghc/master/ghc-1.0-py3-none-any.whl
!cd Wav2Lip && pip install -r requirements.txt
!wget "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -O "C:\Users\307164\Desktop\avatar_chatbot\Wav2Lip\face_detection\detection\sfd\s3fd.pth"