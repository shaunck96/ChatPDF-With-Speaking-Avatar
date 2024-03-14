from sentence_transformers import SentenceTransformer, util
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import csv
import os


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

get_csv("sample_data\dresscode polic- AD.pdf")
