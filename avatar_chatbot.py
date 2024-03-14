import os
import json
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from spire.pdf.common import *
from spire.pdf import *
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
import fitz
import pandas as pd
import re
from unidecode import unidecode
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import string
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
import chromadb
from chromadb.config import Settings
import ast
from nltk.tokenize import sent_tokenize
import chromadb.utils.embedding_functions as embedding_functions
import nltk
from sentence_transformers import SentenceTransformer, util
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
import csv
import os
import subprocess
from base64 import b64encode


#from IPython.display import HTML

nltk.download('punkt')

with open("openai_config.json") as f:
    key = json.load(f)

os.environ["OPENAI_API_KEY"] = key["openai_key"]

def load_preprocess_cluster(file_path):
  doc = fitz.open(file_path)
  for page in doc:
      text = page.get_text()
      print(text)
  output = page.get_text("blocks")
  for page in doc:
      output = page.get_text("blocks")
      previous_block_id = 0 # Set a variable to mark the block id
      for block in output:
          if block[6] == 0: # We only take the text
              if previous_block_id != block[5]:
                  # Compare the block number
                  print("\n")
              print(block[4])
  block_dict = {}
  page_num = 1
  for page in doc: # Iterate all pages in the document
      file_dict = page.get_text('dict') # Get the page dictionary
      block = file_dict['blocks'] # Get the block information
      block_dict[page_num] = block # Store in block dictionary
      page_num += 1 # Increase the page value by 1


  spans = pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax', 'text', 'tag'])
  rows = []
  for page_num, blocks in block_dict.items():
      for block in blocks:
          if block['type'] == 0:
              for line in block['lines']:
                  for span in line['spans']:
                      xmin, ymin, xmax, ymax = list(span['bbox'])
                      font_size = span['size']
                      text = unidecode(span['text'])
                      span_font = span['font']
                      is_upper = False
                      is_bold = False
                      if "bold" in span_font.lower():
                          is_bold = True
                      if re.sub("[\(\[].*?[\)\]]", "", text).isupper():
                          is_upper = True
                      if text.replace(" ","") !=  "":
                          rows.append((xmin, ymin, xmax, ymax, text, is_upper, is_bold, span_font, font_size))
                          span_df = pd.DataFrame(rows, columns=['xmin','ymin','xmax','ymax', 'text', 'is_upper','is_bold','span_font', 'font_size'])

  #span_df['header'] = span_df.groupby(['xmin'])
  scaler = StandardScaler()
  scaled_features = scaler.fit_transform(span_df[['xmin', 'ymin', 'xmax', 'ymax']])

  # DBSCAN clustering
  dbscan = DBSCAN(eps=0.7, min_samples=3)
  clusters = dbscan.fit_predict(scaled_features)

  # Add cluster labels to the DataFrame
  span_df['cluster'] = clusters

  # Print clusters
  for cluster_id in span_df['cluster'].unique():
      print(f"Cluster {cluster_id}:")
      print(span_df[span_df['cluster'] == cluster_id]['text'].values)
      print()

  clustered_pdf_content = pd.DataFrame(span_df.groupby(['cluster'])['text'].apply(lambda x: ''.join(x)))
  return clustered_pdf_content

clustered_pdf_content = load_preprocess_cluster('sample_data\dresscode polic- AD.pdf')
clustered_pdf_content.to_csv(r'clustered_context/clustered_pdf_content.csv', index=False)

def remove_punctuation(input_string):
    # Create a translation table to remove punctuation characters
    translator = str.maketrans('', '', string.punctuation)
    # Apply the translation table to remove punctuation
    return input_string.translate(translator)

def summary_generation(docs, collection_name='AlphaDataFinal'):
  prompt_template = """Write a concise summary of the following:
  "{text}"
  CONCISE SUMMARY:"""
  prompt = PromptTemplate.from_template(prompt_template)

  llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")
  chain = load_summarize_chain(llm, chain_type="stuff")

  return(chain.run([docs]))

def vdb_creation(docs, collection_name):
  doc_and_summary = {}
  for index in range(len(docs)):
    doc_and_summary[index] = {"text": docs[index].page_content, "Summary":summary_generation(docs[index])}

  client = chromadb.Client()

  huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
      api_key="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",
      model_name="sentence-transformers/all-MiniLM-L6-v2"
  )

  collections = client.get_or_create_collection(name=collection_name,
                                                embedding_function=huggingface_ef)

  index = 0
  for i in doc_and_summary.keys():
    sentences = sent_tokenize(doc_and_summary[i]['text'])
    for sentence in sentences:
      print(sentence)
      collections.add(documents = sentence,metadatas = [{"Summary": doc_and_summary[i]['Summary']}],ids = [str(index)])
      index += 1

  return collections

loader = CSVLoader(file_path='clustered_context/clustered_pdf_content.csv')
docs = loader.load()
collection = vdb_creation(docs, "AlphaDataFinal")

def context_retriever(query, collections):
  results = collections.query(query_texts=query,
  #where = sample_where_clause,
  n_results=5)

  context = results['documents']
  return context

#context_retrieved = context_retriever('What specific guidelines does Alpha Data have around dress code?', collection)

def cosine_score_compute(query, qa_dataset, model, threshold=0.6):
    # Encode the query
    query_embedding = model.encode(query)

    # Encode all unique questions in the dataset
    unique_questions = qa_dataset['Question'].unique()
    passages_embeddings = model.encode(unique_questions)

    # Compute cosine similarity scores
    similarity_scores = util.dot_score([query_embedding], passages_embeddings)[0]

    # Filter questions with a similarity score greater than the threshold
    filtered_indices = [i for i, score in enumerate(similarity_scores) if score > threshold]
    filtered_questions = unique_questions[filtered_indices]

    # Merge the filtered questions with the original dataset to get the relevant answers
    relevant_qa_pairs = qa_dataset[qa_dataset['Question'].isin(filtered_questions)]

    # Extract and return the relevant answers
    relevant_answers = relevant_qa_pairs['Answer'].tolist()
    return relevant_answers

def load_llm():
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")
    return llm

def file_processing(file_path):

    # Load data from PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    question_gen = ''

    for page in data:
        question_gen += page.page_content

    splitter_ques_gen = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100
    )

    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)

    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    splitter_ans_gen = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 30
    )

    document_answer_gen = splitter_ans_gen.split_documents(
        document_ques_gen
    )

    return document_ques_gen, document_answer_gen

def llm_pipeline(file_path):

    document_ques_gen, document_answer_gen = file_processing(file_path) #process the file



    llm_ques_gen_pipeline = load_llm() #load the llm

    prompt_template = """
    You are an expert at creating questions based on materials and documentation.
    Your goal is to prepare a set of questions.
    You do this by asking questions about the text below:

    ------------
    {text}
    ------------

    Create questions that will prepare the end users.
    Make sure not to lose any important information.

    QUESTIONS:
    """

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

    refine_template = ("""
    You are an expert at creating questions based on material.
    Your goal is to prepare a set of questions.
    We have received some questions to a certain extent: {existing_answer}.
    We have the option to refine the existing questions or add new ones.
    (only if necessary) with some more context below.
    ------------
    {text}
    ------------

    Given the new context, refine the original questions in English.
    If the context is not helpful, please provide the original questions.
    QUESTIONS:
    """
    )

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    ques_gen_chain = load_summarize_chain(llm = llm_ques_gen_pipeline,
                                            chain_type = "refine",
                                            verbose = True,
                                            question_prompt=PROMPT_QUESTIONS,
                                            refine_prompt=REFINE_PROMPT_QUESTIONS)

    ques = ques_gen_chain.run(document_ques_gen)

    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    llm_answer_gen = load_llm()

    ques_list = ques.split("\n")
    filtered_ques_list = [element for element in ques_list if element.endswith('?') or element.endswith('.')]

    answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen,
                                                chain_type="stuff",
                                                retriever=vector_store.as_retriever())

    return answer_generation_chain, filtered_ques_list

def get_csv (file_path):
    answer_generation_chain, ques_list = llm_pipeline(file_path)
    base_folder = 'qa_dataset/'
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    output_file = base_folder+"QA.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Question", "Answer"])  # Writing the header row

        for question in ques_list:
            print("Question: ", question)
            answer = answer_generation_chain.run(question)
            print("Answer: ", answer)
            print("--------------------------------------------------\n\n")

            # Save answer to CSV file
            csv_writer.writerow([question, answer])
    return output_file

#get_csv("sample_data\dresscode polic- AD.pdf")

model = SentenceTransformer("all-mpnet-base-v2")
qa_dataset = pd.read_csv("qa_dataset/QA.csv")

#print(cosine_score_compute("What is the dress code at alphadata?", qa_dataset, model))


from base64 import b64decode
import numpy as np
from scipy.io.wavfile import read as wav_read
import io
import ffmpeg
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from transformers import pipeline
import torch
from datasets import load_dataset
import soundfile as sf
import numpy as np
import os
import threading

print("\nDone")

import os
import soundfile as sf
import torch
from transformers import pipeline
from datasets import load_dataset, load_from_disk
from base64 import b64encode

class Chatbot:
    def __init__(self):
        self.memory = []
        self.system_prompt = "You are a helpful assistant that helps answers questions about a pdf based on the context provided."
        self.instruction = "Please answer the following question: "
        self.query = ""

        # Initialize models and datasets
        self.chat_model = ChatOpenAI(temperature=0, openai_api_key="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")
        self.synthesizer = pipeline("text-to-speech", model="microsoft/speecht5_tts")
        self.embeddings_dataset = load_dataset(r"C:\Users\307164\Desktop\avatar_chatbot\Matthijs\cmu-arctic-xvectors.py", split="validation")

        # Set file paths
        self.audio_dir = r"C:\Users\307164\Desktop\avatar_chatbot\personna\audio"
        self.wav2lip_dir = r"C:\Users\307164\Desktop\avatar_chatbot"
        self.image_dir = r'C:\Users\307164\Desktop\avatar_chatbot'
        self.response_wav_path = os.path.join(self.audio_dir, "response.wav")
        self.output_video_path = os.path.join(self.wav2lip_dir, "Wav2Lip\result_voice.mp4")

        # Get speaker embedding
        self.audio_embedding = self.get_speaker_embedding(7306)

        # Ensure audio directory exists
        os.makedirs(self.audio_dir, exist_ok=True)

    def get_speaker_embedding(self, index):
        return torch.tensor(self.embeddings_dataset[index]["xvector"]).unsqueeze(0)

    def generate_and_save_audio(self, text):
        result = self.synthesizer(text.content, forward_params={"speaker_embeddings": self.audio_embedding})
        sf.write(self.response_wav_path, result["audio"], samplerate=22050)

    def speech_to_animation(self):
        # Define the command to be run
        command = [
            'python', r'C:\Users\307164\Desktop\avatar_chatbot\Wav2Lip\inference.py',
            '--checkpoint_path', r'C:\Users\307164\Desktop\avatar_chatbot\Wav2Lip\checkpoints\wav2lip_gan.pth',
            '--face', os.path.join(self.audio_dir, 'french_translator_animation.mp4'),
            '--audio', self.response_wav_path
        ]
        
        # Run the command in the Wav2Lip directory
        subprocess.run(command, cwd=self.wav2lip_dir, check=True)
        
        # Read the output video data
        with open(self.output_video_path, 'rb') as f:
            video_data = f.read()
        
        # Encode the video data in base64
        data_url = f"data:video/mp4;base64,{b64encode(video_data).decode()}"
        
        # Return the path to the output video
        return self.output_video_path

    def generate_response(self,
                          user_input,
                          context):
        messages = [
            SystemMessage(
                content=self.system_prompt
            ),
            HumanMessage(
                content=f"{self.instruction} {user_input} {context}"
            ),
        ]

        response = self.chat_model(messages)
        self.generate_and_save_audio(response)
        video_path = self.speech_to_animation()
        print(response)
        return [video_path,response]

    def receive_input(self, user_input):
        self.memory.append({"user": user_input})
        context = cosine_score_compute(user_input, qa_dataset, model)
        response = self.generate_response(user_input, context)
        video_path = response[0]
        self.memory.append({"bot": response[1]})
        return video_path


# Define a function to open a still image using the default image viewer
def open_still_image(image_path):
    # This will open the image in the default viewer of your operating system
    os.startfile(image_path) if os.name == 'nt' else subprocess.call(['open', image_path])
    print("Bot is typing...")

def open_still_image(image_path):
    os.startfile(image_path)

# Define a function to open video response
def open_video(video_path):
    # This will open the video in the default video player of your operating system
    os.startfile(video_path) if os.name == 'nt' else subprocess.call(['open', video_path])

import streamlit as st

bot = Chatbot()

# Streamlit page configuration
st.set_page_config(page_title="PDF Contextual Chatbot", layout="wide")

# UI setup
st.title("PDF Contextual Chatbot")

# Initialize a placeholder for video or image
video_placeholder = st.empty()

# Initially display the static image in the placeholder
video_placeholder.image("french_translator.PNG", caption="AlphaBot")

# User input
user_input = st.text_input("Type your question here and press Enter:")

if user_input:
    with st.spinner('Thinking...'):
        # Process the input and get the response
        video_path = bot.receive_input(user_input)

        if video_path:
            # Update the placeholder to display the video instead of the image
            video_placeholder.video(video_path)
        else:
            # Optionally, handle the case where there is no video response.
            # For now, we'll just keep showing the original image.
            pass

#interactive_chat_bot()
