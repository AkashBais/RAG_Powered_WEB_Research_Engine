from tqdm.notebook import tqdm
import pandas as pd
from typing import Optional, List, Tuple
from datasets import Dataset
import matplotlib.pyplot as plt

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument

from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

from langchain.llms import CTransformers
from ctransformers import AutoModelForCausalLM as AutoModelForCausalLM_CT


from sentence_transformers import SentenceTransformer

from typing import Union
import os, sys
sys.path.append(r"./Test_RAG_ChatBot/")
sys.path.append(r"/content/RAG/Test_RAG_ChatBot/")

import config

### importing custome packages
from src.create_data_chunks_utils import split_documents
from src.extract_data_utils import search_web


from transformers import pipeline
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig

import pacmap
import numpy as np
import plotly.express as px

import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown

from dotenv import load_dotenv

import concurrent.futures

import re

# Used to securely store your API key
from google.colab import userdata

from transformers import LlamaTokenizer
# GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')

load_dotenv(config.DOT_ENV_PATH)


genai.configure(api_key= os.environ.get("GOOGLE_API_KEY") )

class execute_rag():
  def __init__( 
      self,
      model = None,
      tokenizer = None,
      path:str=None,
      save_model_locally:bool=True,
      gemini:bool=False,
      pre_quantized:bool=False,
      verbos:int=0
      )-> None:

    '''

    Init method performs the following tasks:
    a.) Initilize the LLM to be used in RAG
    b.) Initilize the tokenizer to be used in RAG
    c.) Initilize a "text-generation" Hugging Face pipeline using the above LLM and Tokenizer

    Params:
    model = None -> Instance of the LLM model to use, If None will load the appropriate model based on the config file
    tokenizer = None -> Instance of the LLM tokenizer to use, If None will load the appropriate tokenizer based on the config file
    path:str=None -> Path to a locally stored model weights
    save_model_locally:bool=True -> Bool flag deciding wether to store the weights of the loaded model.Saved to a path based on config file 
    gemini:bool=False -> Boolean flag to use gemini api as an LLM for the RAG [In cases we don't have enough compute to runn an LLM]
    pre_quantized:bool=False -> Do we want to load a prequantized model, If true will load a GGUF version of Llama-2 7B 
    # TO-DO: The model choice for pre_quantized model is currently hard coded due to support reasions.Broaden the same
    verbos:int=0  -> Do we want to enable verbos while exicuting
    '''   
    self.gemini = gemini
    self.pre_quantized = pre_quantized

    self.embedding_projector = None
    self.documents_projected = None
    self.verbos = verbos

    if model is None or tokenizer is None:
      
      if not self.gemini:
        
        if not pre_quantized:
          print(f"Initilizing reader LLM as {config.READER_MODEL_NAME}")
          print(f"Initilizing tokenizer as {config.READER_MODEL_NAME}")
          self.tokenizer = AutoTokenizer.from_pretrained(config.READER_MODEL_NAME,truncation=True)

          if path is None:
            if torch.cuda.is_available():
              bnb_config = BitsAndBytesConfig(
                  load_in_4bit=True,
                  bnb_4bit_use_double_quant=True,
                  bnb_4bit_quant_type="nf4",
                  bnb_4bit_compute_dtype=torch.bfloat16 
              )


            else:
              bnb_config = None

            model = AutoModelForCausalLM.from_pretrained(config.READER_MODEL_NAME,
                                                          quantization_config=bnb_config,
                                                          trust_remote_code=True,
                                                          device_map="auto")


            if save_model_locally:
              print("Saving model")
              self.path = os.path.join(config.MODEL_FOLDER_PATH,config.READER_MODEL_NAME)
              os.makedirs(self.path, exist_ok=True) 
              model.save_pretrained(self.path, from_pt=True)


          else:
            model = AutoModelForCausalLM.from_pretrained(path,
                                                    use_safetensors=True,
                                                    local_files_only=True,
                                                    device_map="auto",
                                                    trust_remote_code=True,)
        else:
            
          print(f"Initilizing reader LLM as 'TheBloke/Llama-2-7B-GGUF'")
          print(f"Initilizing tokenizer as 'hf-internal-testing/llama-tokenizer'")   
          model = AutoModelForCausalLM_CT.from_pretrained("TheBloke/Llama-2-7B-GGUF", hf=True,context_length=3000)
          from transformers import LlamaTokenizerFast
          self.tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer",truncation=True,model_max_length = 3000,legacy = False)
             
        self.READER_LLM = pipeline(
        model=model,
        tokenizer = self.tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens = 512,

        )


      else:
        print(f"Initilizing reader LLM as {config.READER_LLM_GIMINI}")
        self.READER_LLM = genai.GenerativeModel(config.READER_LLM_GIMINI)
      
    else:
      print(f"Initilizing reader LLM and tokenizer as the passed objects")
      self.READER_LLM = model
      self.tokenizer = tokenizer
      
    print("Reader LLM initilization complet")

  def remove_elements(self,text):
      '''
      This is a helper function to clean text
      '''
      # Remove number followed by ')' or '.'
      text = re.sub(r'\d+[.)]', '', text)
      # Remove individual occurrences of '.' and ')'
      text = text.replace('.', '').replace(')', '').replace('-', '')
      return text

  def combine_documents(self,doc_list):
      '''
      This is a helper function to combine all Langchain Document list together
      '''
      combined_doc = []
      for doc in doc_list:
          combined_doc += doc
      return combined_doc

  def init_knowledge_base(
      self,
      web_search_term:str,
      engine:str="google",
      chunk_size:int=512,
      chunk_overlap: Optional[int] = None,
      )-> None:

    '''
    This function initilizes the knowledge base to be used in RAG. Will scrape internet data to genrate relevant knowledge base for RAG

    Params:
    web_search_term:str -> The query you want to search over the web
    engine:str="google" -> The web search engine to use
    chunk_size:int=512  -> Maximum size of the individual chunks in intiger 
    chunk_overlap: Optional[int] = None -> Over lap between chunks, If None it is computed as int(chunk_size/10)

    '''
    print("Creating Knowledge base")
 

    try:

      print("Genrating web queries")

      '''
      We use Google's GEMINI MODEL API to genrate a brief background of the web_search_term pulling from it's web knowledgebase
      '''
      model = genai.GenerativeModel(config.GEMINI_MODEL)
      response = model.generate_content(config.PROMPT_WEB_SEARCH_FORMAT.format(web_search_term,web_search_term,web_search_term,web_search_term,web_search_term,web_search_term,web_search_term))
      summary = response.candidates[0].content.parts[0].text
      '''
      We use Google's GEMINI MODEL API to genrate a 3 more relevant web search terms similar to our search term using the brief background
      fetched earlier
      '''

      response = model.generate_content(config.PROMPT_GENRATE_WEB_QUERY_FORMAT.format(summary))
      query_list = response.candidates[0].content.parts[0].text.split("\n")
      result = [self.remove_elements(ele).strip() for ele in query_list if len(ele) > 0] + [web_search_term]
    except:
      print(f"Please make the following improvements to the query: /n {query_list}")

    search_result = []



    '''
    We run concurent threads for each API call to the search engin's API
    '''
    data_temp = [ ( web_search_term,engine) for web_search_term in result ]
    with concurrent.futures.ThreadPoolExecutor() as executor:
      search_result = list(executor.map(lambda params: search_web(*params), data_temp))

    search_result = self.combine_documents(search_result)
 
    print(f"Processing search results")
    '''
    Creating semantic chunks from the LangChain document list returned by the search_web call
    '''
    self.docs_processed = split_documents(
      chunk_size = chunk_size,
      knowledge_base = search_result,
      chunk_overlap = chunk_overlap,
      )


    model_kwargs = {"device": 'cuda:0' if torch.cuda.is_available() else 'cpu'}

    print(f"Embedding search results in a vector DB")
    '''
    Initilizing an embedding model to be used in the vectorization of the LangChain Documents 
    '''
    self.embedding_model = HuggingFaceEmbeddings(
      model_name = config.EMBEDDING_MODEL_NAME,
      multi_process = True,
      model_kwargs =model_kwargs,
      encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
    )

    '''
    Initilizing the vector database. We use Facebook AI Similarity Search (Faiss) as a vector DB
    '''
    self.VECTOR_DATABASE = FAISS.from_documents(
        self.docs_processed,
        self.embedding_model,
        distance_strategy=DistanceStrategy.COSINE
    )

    print(f"Vector DB created")

    del search_result

  

  def query(
      self,
      user_query:str,
      document_fetch_vectorstore:int=5,
      document_fetch_similaritysearch:int=1,
      explanablity:bool=True,
      ) -> str:

    '''
    This method lets user query the application based on the database initilized using init_knowledge_base method

    user_query:str -> Query for the RAG application
    document_fetch_vectorstore:int=5 -> Document to fetch from the Vector DB [For MMR]
    document_fetch_similaritysearch:int=1 -> Documents to return amongst th efetched documents [For MMR]
    explanablity:bool=True -> Do we want to plot a 2D projection of Query and vector DB for better understanding
    '''
    print("Embedding user query")
    query_vector = self.embedding_model.embed_query(user_query)

    print(f"\nStarting retrieval for {user_query=}...")

    try:
        print("Genrating similar queries")
        model = genai.GenerativeModel(config.GEMINI_MODEL)
        '''
        We use Google's GEMINI MODEL API to genrate a 3 more relevant retrival terms similar to our orignal retrival term 
        '''        
        response = model.generate_content(config.PROMPT_GENRATE_VECTOR_DB_QUERY_FORMAT.format(user_query))
        query_list = response.candidates[0].content.parts[0].text.split("\n")
        result = [self.remove_elements(ele).strip() for ele in query_list if len(ele) > 0] + [user_query]
        print(result) 
    except:
      print(f"Please make the following improvements to the query: /n {query_list}")

    data_temp = [ ( web_search_term,document_fetch_similaritysearch,document_fetch_vectorstore) for web_search_term in result ]
    with concurrent.futures.ThreadPoolExecutor() as executor:
      retrieved_documents = list(executor.map(lambda params: self.VECTOR_DATABASE.max_marginal_relevance_search(user_query,k=document_fetch_similaritysearch, fetch_k=document_fetch_vectorstore), data_temp))

    

    retrieved_documents = self.combine_documents(retrieved_documents)


    retrieved_documents_text = [document.page_content for document in retrieved_documents]
    retrieved_documents_source = [document.metadata["source"] for document in retrieved_documents]
    context = "\nExtracted documents:\n"

    context += "".join([f"Document {str(i)}:::\n" + document for i,document in enumerate(retrieved_documents_text)])

    
    print(f"\nCompiling responce for {user_query=}...")
    if not self.gemini:
      RAG_PROMPT_TEMPLATE = self.tokenizer.apply_chat_template(
        config.PROMPT_IN_CHAT_FORMAT, tokenize=False, add_generation_prompt=True
      )

      final_prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=user_query)
      if self.verbos == 1:
        print(f"final_prompt: {final_prompt}\nfinal_prompt_len:{len(self.tokenizer.encode(final_prompt))}\nmodel_max_length :{self.tokenizer.model_max_length}")
      # print(f"final_prompt_len:{len(self.tokenizer.encode(final_prompt))}\nmodel_max_length :{self.tokenizer.model_max_length}")
      raw_output = self.READER_LLM(final_prompt)
      answer = raw_output[0]["generated_text"]
    
    else:
      def gimini_rag_prompt(query, context):
        escaped = context.replace("\n", " ")#.replace("'", "").replace('"', "")
        prompt = (config.GIMINI_RAG_PROMPT).format(query=query, context=escaped)

        return prompt

      answer = self.READER_LLM.generate_content(gimini_rag_prompt(user_query, context)).text

    if explanablity:
      print("Ploting for Explanability")
      if self.embedding_projector is None or self.documents_projected is None:
      
        self.embedding_projector = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=1)

        self.embeddings_2d = [
            list(self.VECTOR_DATABASE.index.reconstruct_n(idx, 1)[0]) for idx in range(len(self.docs_processed))
        ]
        # Fit the data (the index of transformed data corresponds to the index of the original data)
        self.documents_projected = self.embedding_projector.fit_transform(np.array(self.embeddings_2d + [query_vector]), init="pca")
      
      df = pd.DataFrame.from_dict(
          [
              {
                  "x": self.documents_projected[i, 0],
                  "y": self.documents_projected[i, 1],
                  "source": self.docs_processed[i].metadata["source"].split("//")[1].split("/")[0],
                  "extract": self.docs_processed[i].page_content[:200] + "...",
                  "symbol": "circle",
                  "size_col": 4,
              }
              for i in range(len(self.docs_processed))
          ]
          + [
              {
                  "x": self.documents_projected[-1, 0],
                  "y": self.documents_projected[-1, 1],
                  "source": "User query",
                  "extract": user_query,
                  "size_col": 100,
                  "symbol": "star",
              }
          ]
      )

      # Visualize the embedding
      fig = px.scatter(
          df,
          x="x",
          y="y",
          color="source",
          hover_data="extract",
          size="size_col",
          symbol="symbol",
          color_discrete_map={"User query": "black"},
          width=1000,
          height=700,
      )
      fig.update_traces(
          marker=dict(opacity=1, line=dict(width=0, color="DarkSlateGrey")),
          selector=dict(mode="markers"),
      )
      fig.update_layout(
          legend_title_text="<b>Chunk source</b>",
          title="<b>2D Projection of Chunk Embeddings via PaCMAP</b>",
      )
      fig.show()

      del df
    return answer,retrieved_documents_source,raw_output








    