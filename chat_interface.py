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

bot = Chatbot()
# Set up Streamlit session state to store conversation
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Start a form for user message input
with st.form("user_input_form"):
    # Text input for the user message within the form
    user_input = st.text_input("Message:", key="user_input", placeholder="Type your message here...")

    # Button to send the message within the form
    submitted = st.form_submit_button('Send')

# Logic to handle when the message is sent
if submitted:
    if user_input.lower().strip() in ["thank you", "that's all for now", "goodbye"]:
        st.session_state.conversation.append(("You", user_input))
        st.session_state.conversation.append(("Bot", "You're welcome! Feel free to ask me anything anytime."))
        st.info("Conversation ended. Refresh the page to start a new conversation.")
    else:
        # Add user message to the conversation
        st.session_state.conversation.append(("You", user_input))

        # Generate and add bot response to the conversation
        response = bot.receive_input(user_input)
        st.session_state.conversation.append(("Bot", response))

# Display the conversation in a fancier way
st.write("### Conversation")
for author, message in st.session_state.conversation:
    # Check the author to apply different styling
    if author == "You":
        # Display user messages on the left
        st.container().markdown(f"**You**: {message}", unsafe_allow_html=True)
    else:
        # Display bot messages on the right with a different color
        st.container().markdown(f"<div style='background-color:#f0f2f6;padding:10px;border-radius:10px;margin-bottom:10px;'>**Bot**: {message}</div>", unsafe_allow_html=True)
