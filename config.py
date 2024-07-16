DATA_FOLDER_PATH = "./RAG/Test_RAG_ChatBot/data"
MODEL_FOLDER_PATH = "./RAG/Test_RAG_ChatBot/models"
DOT_ENV_PATH = "/content/RAG/Test_RAG_ChatBot/keys.env"


DATA_FILE_NAME = ["GAN.pdf"]
SEPARATORS = ["\n\n", "\n", ".", ""]
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta" #"TheBloke/Llama-2-7B-GGUF"#"TheBloke/zephyr-7B-beta-GGUF"#"marella/gpt-2-ggml"# "TheBloke/Llama-2-7B-Chat-GGML"## "marella/gpt-2-ggml"#"TheBloke/Llama-2-7B-Chat-GGUF"#
# READER_MODEL_TOKANOZER =  "HuggingFaceH4/zephyr-7b-alpha"# "gpt2" #"TheBloke/Llama-2-7B-Chat-GGML"# "TheBloke/Llama-2-7B-Chat-GGUF"#"gpt2"
# READER_MODEL_NAME="microsoft/Phi-3-small-128k-instruct"
GEMINI_MODEL = 'models/gemini-1.5-flash-latest'#'Gemini 1.5 Pro'#'models/gemini-1.5-flash-latest'
READER_LLM_GIMINI = 'models/gemini-1.5-flash-latest'#'Gemini 1.5 Pro'
# READER_MODEL_NAME="microsoft/Phi-3-mini-128k-instruct"
# READER_MODEL_NAME="microsoft/Phi-3-mini-4k-instruct"

PROMPT_WEB_SEARCH_FORMAT = """
Please provide a comprehensive summary about the search term {} to get a better and broader understanding of the term.
The summary should include the following points
1 Background: Provide a general overview and context about {}.
2 Brief History: Discuss the origin and evolution of {} over time.
3 Market Landscape: Analyze the current market scenario, trends, and dynamics related to {}.
4 Competitors: Identify and describe the main competitors in the {} space.
5.Future scope: Analyze the plausible future market scenario, trends, and dynamics related to {}.
6.Implications: Analyze the implications of {} in tearms of the current market landscape"""   

PROMPT_GENRATE_WEB_QUERY_FORMAT ="""
Given the following summary, strictly identify the most relevent 3 search terms that are diverse and are best to search over the internet to explore the topic broadly:

summary:{}

The search terms should cover different aspects of the topic to ensure a comprehensive understanding.
Strictly return the search terms and abslutely nothing else in the responce.Don't return any metadata or discription or explanation.
Sample genration for search term 'Novartis' looks like this 'Novartis market landscape'\n'Novartis competitors'\n'Novartis history and mergers'. 
\n is the seprators between genrated search terms
"""

PROMPT_GENRATE_VECTOR_DB_QUERY_FORMAT ="""
Given the following query, identify the top 3 addational diverse queries that can also be searched in the vector database to explore the topic broadly:

query:{}

The addational queries should cover different aspects of the topic to ensure a comprehensive understanding.
Strictly return the search terms and abslutely nothing else in the responce.No need to return any metadata or discription or explanation.
Sample genration for queries 'Describe Novartis' looks like this 'Novartis market landscape'\n'Novartis competitors'\n'Novartis history and mergers'. 
\n is the seprators between search terms
"""

PROMPT_IN_CHAT_FORMAT = [
    {
        "role": "system",
        "content": """Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.""",
    },
    {
        "role": "user",
        "content": """ 
Use the following Context to answer the question.
Context:{context}
---------
Here is the question you need to answer.
Question: {question}
""",
    },
]


GIMINI_RAG_PROMPT = """You are a helpful and informative bot that answers questions using text from the reference passage included below. \
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
Respond only to the question asked, response should be concise and relevant to the question.\
Strike a friendly and converstional tone. \
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer..
QUESTION: '{query}'
CONTEXT: '{context}'

ANSWER:
"""
# PROMPT_IN_CHAT_FORMAT = """<s>[INST] <<SYS>> {Use the following pieces of context to answer the question at the end.
# If you don't know the answer, just say that you don't know, don't try to make up an answer. 
# Don't include any text that isn't an answer to the question.}
# <</SYS>>

# {context}

# Question: {question}
# Answer:"""

# PROMPT_IN_CHAT_FORMAT = ['''<s>[INST] <<SYS>>
# {your_system_message}
# <</SYS>>

# {user_message_1} [/INST] {model_reply_1}</s><s>[INST] {user_message_2} [/INST]''']
