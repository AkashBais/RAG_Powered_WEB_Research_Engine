import config
from typing import Union


from tqdm.notebook import tqdm
import pandas as pd
from typing import Optional, List, Tuple
from datasets import Dataset
import matplotlib.pyplot as plt

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(
    chunk_size: int,
    knowledge_base: Union[List[LangchainDocument], LangchainDocument],
    chunk_overlap: Optional[int] = None,
    tokenizer_name: Optional[str] = config.EMBEDDING_MODEL_NAME,
    visvalize_token_distribution: Optional[bool] = True,
) -> List[LangchainDocument]:

  '''
  chunk_size: int -> Maximum size of the individual chunks in intiger
  knowledge_base: Union[List[LangchainDocument], LangchainDocument] -> Single of a list of LangChain Documents to chunk
  chunk_overlap: Optional[int] = None -> Over lap between chunks, If None it is computed as int(chunk_size/10)
  tokenizer_name: Optional[str] = config.EMBEDDING_MODEL_NAME -> Model ID of the embedding model to use
  visvalize_token_distribution: Optional[bool] = True -> If true the below code helps us visvalize the size of created chunks
  in terms of tokens
  '''
  
  if chunk_overlap is None:
    chunk_overlap = int(chunk_size / 10)

  '''
  Text splitter that uses HuggingFace tokenizer to count length.
  '''


  # Initialize tokenizer
  text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
      AutoTokenizer.from_pretrained(tokenizer_name),
      chunk_size = chunk_size,
      chunk_overlap = chunk_overlap,
      add_start_index = True,
      strip_whitespace=True,
      separators = config.SEPARATORS,
                                                                            
  )

  '''
  Applying text splitter to the LandDocuments in the knowledge base
  '''
  unique_tests = {}
  docs_processed = []

  for document in knowledge_base:
    docs_processed += text_splitter.split_documents([document])

  docs_processed_unique = []
  '''
  Checking and deduplicating the contents of the chunks created. 
  Might take longer for big production level documents. Can be removed
  ''' 
  for document in docs_processed:
    if document.page_content not in unique_tests:
      unique_tests[document.page_content] = True
      docs_processed_unique.append(document)


  '''
  if visvalize_token_distribution pram is true the below code helps us visvalize the size of created chunks
  in terms of tokens. This is intended to inform:
  a.) Is the size of the chunks created in range of teh embedding model context size
  b.) How the chunk size distribution looks. Are the chinks predomintely small, is so individual chunks might lack relevant context
  '''   
  if visvalize_token_distribution:
    tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL_NAME)
    lengths = [len(tokenizer.encode(doc.page_content)) for doc in docs_processed]
    print(f"Model's maximum sequence length: {SentenceTransformer(config.EMBEDDING_MODEL_NAME).max_seq_length}")
    print(f"Chunks's maximum sequence length: {max(lengths)}")
    fig = pd.Series(lengths).hist()
    plt.title("Distribution of document lengths in the knowledge base (in count of tokens)")
    plt.show()



  return docs_processed_unique