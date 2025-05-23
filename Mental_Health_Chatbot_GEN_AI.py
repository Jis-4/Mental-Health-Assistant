#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.system('pip install langchain_groq langchain_core langchain_community')


# In[3]:


from langchain_groq import ChatGroq
llm = ChatGroq(
    temperature = 0,
    groq_api_key = "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",
    model_name = "llama-3.3-70b-versatile"
)
result = llm.invoke("Who is lord Ram?")
print(result.content)


# In[4]:

os.system('pip install pypdf')
get_ipython().system('pip install pypdf')


# In[5]:

os.system('pip install chromadb')
get_ipython().system('pip install chromadb')


# In[6]:

os.system('pip install sentence_transformers')
get_ipython().system('pip install sentence_transformers')


# In[11]:


pip install chromadb


# In[12]:


from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
def initialize_llm():
  llm = ChatGroq(
    temperature = 0,
    groq_api_key = "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",
    model_name = "llama-3.3-70b-versatile"
)
  return llm

def create_vector_db():
  loader = DirectoryLoader(r"C:\Users\Kalash bhadoriya\OneDrive\Documents\mental health chatbot\data", glob='*.pdf', loader_cls=PyPDFLoader)
  documents = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
  texts = text_splitter.split_documents(documents)
  embeddings = HuggingFaceBgeEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
  vector_db = Chroma.from_documents(texts, embeddings, persist_directory = './chroma_db')
  vector_db.persist()

  print("ChromaDB created and data saved")

  return vector_db

def setup_qa_chain(vector_db, llm):
  retriever = vector_db.as_retriever()
  prompt_templates = """ You are a compassionate mental health chatbot. Respond thoughtfully to the following question:
    {context}
    User: {question}
    Chatbot: """
  PROMPT = PromptTemplate(template = prompt_templates, input_variables = ['context', 'question'])

  qa_chain = RetrievalQA.from_chain_type(
      llm = llm,
      chain_type = "stuff",
      retriever = retriever,
      chain_type_kwargs = {"prompt": PROMPT}
  )
  return qa_chain


def main():
  print("Intializing Chatbot.........")
  llm = initialize_llm()

  db_path = "/content/chroma_db"

  if not os.path.exists(db_path):
    vector_db  = create_vector_db()
  else:
    embeddings = HuggingFaceBgeEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
  qa_chain = setup_qa_chain(vector_db, llm)

  while True:
    query = input("\nHuman: ")
    if query.lower()  == "exit":
      print("Chatbot: Take Care of yourself, Goodbye!")
      break
    response = qa_chain.run(query)
    print(f"Chatbot: {response}")

if __name__ == "__main__":
  main()


# In[13]:

os.system('pip install gradio')
get_ipython().system('pip install gradio')


# In[17]:


import os

# Disable the symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import gradio as gr

def initialize_llm():
  llm = ChatGroq(
    temperature = 0,
    groq_api_key = "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",
    model_name = "llama-3.3-70b-versatile"
)
  return llm

def create_vector_db():
  loader = DirectoryLoader(r"C:\Users\Kalash bhadoriya\OneDrive\Documents\mental health chatbot\data", glob = '*.pdf', loader_cls = PyPDFLoader)
  documents = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
  texts = text_splitter.split_documents(documents)
  embeddings = HuggingFaceBgeEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
  vector_db = Chroma.from_documents(texts, embeddings, persist_directory = './chroma_db')
  vector_db.persist()

  print("ChromaDB created and data saved")

  return vector_db

def setup_qa_chain(vector_db, llm):
  retriever = vector_db.as_retriever()
  prompt_templates = """ You are a compassionate mental health chatbot. Respond thoughtfully to the following question:
    {context}
    User: {question}
    Chatbot: """
  PROMPT = PromptTemplate(template = prompt_templates, input_variables = ['context', 'question'])

  qa_chain = RetrievalQA.from_chain_type(
      llm = llm,
      chain_type = "stuff",
      retriever = retriever,
      chain_type_kwargs = {"prompt": PROMPT}
  )
  return qa_chain


print("Intializing Chatbot.........")
llm = initialize_llm()

db_path = "/content/chroma_db"

if not os.path.exists(db_path):
  vector_db  = create_vector_db()
else:
  embeddings = HuggingFaceBgeEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
  vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
qa_chain = setup_qa_chain(vector_db, llm)

def chatbot_response(query):
    # Your logic to get the response from the chatbot
    response = qa_chain.run(query)
    return [{"role": "user", "content": query}, {"role": "assistant", "content": response}]

iface = gr.Interface(
    fn=chatbot_response,
    inputs="text",
    outputs=gr.Chatbot(type="messages"),
    live=True
)

iface.launch()


# In[ ]:




