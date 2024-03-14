from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from nltk.tokenize import sent_tokenize



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
collections = vdb_creation(docs, "AlphaDataFinal")
results = collections.query(query_texts="Dress Code for Men",
#where = sample_where_clause,
n_results=2)

context = results['documents']
print(context)
