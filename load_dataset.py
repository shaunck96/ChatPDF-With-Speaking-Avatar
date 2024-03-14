from datasets import load_dataset
import torch 

embeddings_dataset = load_dataset(r"C:\Users\307164\Desktop\avatar_chatbot\Matthijs\cmu-arctic-xvectors.py", split="validation")

speaker_embeddings = embeddings_dataset[7306]["xvector"]
speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)
speaker_embeddings