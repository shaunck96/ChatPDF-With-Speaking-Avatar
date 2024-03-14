from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from nltk.tokenize import sent_tokenize
from langchain_community.document_loaders.csv_loader import CSVLoader
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from nltk.tokenize import sent_tokenize
import fitz
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import json
import os
import re
from unidecode import unidecode
from datasets import load_dataset
import torch
from transformers import pipeline
import soundfile as sf
import subprocess

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
    clustered_pdf_content.to_csv(r'clustered_context\clustered_pdf_content.csv', index=False)


def summary_generation(docs, collection_name='AlphaDataFinal'):
    prompt_template = """Write a concise summary of the following describing keywords and key points discussed:
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


class Chatbot():
    def __init__(self):
        self.memory = []
        self.system_prompt = "You are a helpful assistant that helps answers questions about a pdf based on the context provided."
        self.instruction = "Please answer the following question in less than 40 words: "
        self.query = ""
        self.chat_model = ChatOpenAI(temperature=0, openai_api_key="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")
        self.model = SentenceTransformer("all-mpnet-base-v2")
        load_preprocess_cluster(r'sample_data\Natural+Language+Processing+-+Test+2+about+model+optimization.pdf')
        self.loader = CSVLoader(file_path=r'clustered_context\clustered_pdf_content.csv')
        self.docs = self.loader.load()
        self.collection = vdb_creation(self.docs, "AlphaDataFinal")
        self.embeddings_dataset = load_dataset(r"C:\Users\307164\Desktop\avatar_chatbot\Matthijs\cmu-arctic-xvectors.py", split="validation")
        self.audio_embedding = self.get_speaker_embedding(self.embeddings_dataset, 7306)
        self.synthesizer = pipeline("text-to-speech", model="microsoft/speecht5_tts")

    def get_speaker_embedding(self, embeddings_dataset, index):
        return torch.tensor(embeddings_dataset[index]["xvector"]).unsqueeze(0)

    def speech_generation(self, response):
        result = self.synthesizer(response, forward_params={"speaker_embeddings": self.audio_embedding})
        sf.write(r"C:\\Users\\307164\\Desktop\\avatar_chatbot\\personna\\audio\\response.wav", result["audio"], samplerate=22050)
        command = 'cd Wav2Lip && python "C:\\Users\\307164\\Desktop\\avatar_chatbot\\Wav2Lip\\inference.py" --checkpoint_path "C:\\Users\\307164\\Desktop\\avatar_chatbot\\Wav2Lip\\checkpoints\\wav2lip_gan.pth" --face "C:\\Users\\307164\\Desktop\\avatar_chatbot\\personna\\audio\\french_translator_animation.mp4" --audio "C:\\Users\\307164\\Desktop\\avatar_chatbot\\personna\\audio\\response.wav"'
        subprocess.run(command, shell=True)


    def context_retriever(self, query):
        results = self.collection.query(query_texts=query,
        #where = sample_where_clause,
        n_results=5)

        context = results['documents']
        return context

    def cosine_score_compute(self, query, qa_dataset, model, threshold=0.6):
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
        relevant_answers = ', '.join(relevant_qa_pairs['Answer'].tolist())
        return relevant_answers
    

    def generate_response(self,
                          user_input,
                          context):
        messages = [
            SystemMessage(
                content=self.system_prompt
            ),
            HumanMessage(
                content=f"{self.instruction} User Query: {user_input} Context to answer the question: {context}"
            ),
        ]

        response = self.chat_model(messages)
        return response

    def receive_input(self, user_input):
        self.memory.append({"user": user_input})
        qa_dataset = pd.read_csv(r'qa_dataset\QA.csv')
        #context = self.cosine_score_compute(user_input, qa_dataset, self.model)
        context = self.context_retriever(user_input)
        response = self.generate_response(user_input, context)
        self.memory.append({"bot": response})
        self.speech_generation(response.content)
        return response.content


import streamlit as st
import base64
from streamlit_player import st_player  # Make sure to install streamlit-player for video embedding

# Custom CSS to inject contained in a string
custom_css = """
    <style>
        /* Main page layout */
        .reportview-container .main .block-container{
            max-width: 800px;
            padding-top: 5rem;
            padding-right: 1rem;
            padding-left: 1rem;
            padding-bottom: 5rem;
        }

        /* Chat message styling */
        .stMarkdown {
            background-color: #f0f2f6;
            border-radius: 20px;
            padding: 10px;
            margin-bottom: 20px;
        }

        /* Input field styling */
        .stTextInput>div>div>input {
            border-radius: 20px;
        }

        /* Button styling */
        .stButton>button {
            border-radius: 20px;
            border: 1px solid #4CAF50;
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            cursor: pointer;
            width: 100%;
        }

        .stButton>button:hover {
            background-color: #45a049;
        }

        /* Video container styling */
        .video-container {
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            transition: 0.3s;
            border-radius: 10px;
        }

        .video-container:hover {
            box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
        }
    </style>
"""

# Inject custom CSS with Markdown
st.markdown(custom_css, unsafe_allow_html=True)

# Assuming the Chatbot class is defined elsewhere
if 'bot' not in st.session_state:
    st.session_state['bot'] = Chatbot()

if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

if 'current_video' not in st.session_state:
    st.session_state['current_video'] = 'intro'

if 'video_queue' not in st.session_state:
    st.session_state['video_queue'] = []

# Function to generate a base64 encoding of the video
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to display a video
def show_video(file_path, loop=False, autoplay=True, muted=True, controls=True, height=400):
    video_encoded = get_base64_of_bin_file(file_path)
    loop_attr = "loop" if loop else ""
    autoplay_attr = "autoplay" if autoplay else ""
    muted_attr = "muted" if muted else ""
    controls_attr = "controls" if controls else ""
    video_html = f'''
    <div class="video-container" style="max-width: 100%; margin: auto; overflow: hidden;">
        <video {loop_attr} {autoplay_attr} {muted_attr} {controls_attr} style="max-width: 100%; height: auto; object-fit: contain; border-radius: 20px;">
            <source src="data:video/mp4;base64,{video_encoded}" type="video/mp4" />
        </video>
    </div>
    '''
    st.components.v1.html(video_html, height=height)

# Include the custom CSS we defined earlier
st.markdown(custom_css, unsafe_allow_html=True)

# Function to manage video transitions
def update_video_area():
    if st.session_state['current_video'] == 'intro':
        show_video(r'C:\Users\307164\Desktop\avatar_chatbot\personna\intro.mp4', loop=False)
    elif st.session_state['current_video'] == 'filler':
        show_video(r'C:\Users\307164\Desktop\avatar_chatbot\personna\filler.mp4', loop=True)
    elif st.session_state['current_video'] == 'response' and st.session_state['video_queue']:
        video_path = st.session_state['video_queue'].pop(0)  # Get the first video from the queue
        show_video(video_path, loop=False)
    elif st.session_state['current_video'] == 'follow_up':
        show_video(r'C:\Users\307164\Desktop\avatar_chatbot\personna\filler.mp4', loop=False)
    # Update the current video state based on the queue
    if st.session_state['video_queue']:
        st.session_state['current_video'] = 'response'
    else:
        st.session_state['current_video'] = 'filler'

# Initial video playback
if st.session_state['current_video'] == 'intro':
    update_video_area()

# Include the custom CSS we defined earlier
st.markdown(custom_css, unsafe_allow_html=True)


# Chat Area
st.write("### Conversation")
chat_container = st.container()
with chat_container:
    for author, message in st.session_state['conversation']:
        # Use different background colors for user and bot
        if author == "You":
            st.markdown(f"<div style='background-color: #e1f5fe; border-radius: 20px; padding: 10px;'>**{author}**: {message}</div>", unsafe_allow_html=True)
        else:  # Bot's messages
            st.markdown(f"<div style='background-color: #ede7f6; border-radius: 20px; padding: 10px;'>**{author}**: {message}</div>", unsafe_allow_html=True)

# Handling user input
with st.form("user_input_form"):
    user_input = st.text_input("Message:", placeholder="Type your message here...")
    submitted = st.form_submit_button('Send')

if submitted:
    # Simulate processing user message and generating a response
    response = st.session_state['bot'].receive_input(user_input)  # Adapt this with your logic
    st.session_state['conversation'].append(("You", user_input))
    st.session_state['conversation'].append(("Bot", response))
    # Example path, adapt with logic to choose the correct video based on the response
    st.session_state['video_queue'].append(r'C:\Users\307164\Desktop\avatar_chatbot\Wav2Lip\results\result_voice.mp4')
    st.session_state['current_video'] = 'response'
    update_video_area()

    # Queue a follow-up question video (adjust path as necessary)
    st.session_state['video_queue'].append(r'C:\Users\307164\Desktop\avatar_chatbot\personna\filler.mp4')

# Scroll to the last message
if st.session_state['conversation']:
    st.script_request_queue.enqueue('scrollTo', {'id': chat_container._block_hash})
# Conversation UI
st.write("### Conversation")
for author, message in st.session_state['conversation']:
    st.markdown(f"**{author}**: {message}")

