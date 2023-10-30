from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import csv_loader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings

## generate llm model
llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGUF", model_file="llama-2-7b-chat.Q5_K_M.gguf", model_type="llama")

loader = csv_loader.CSVLoader(
    file_path="./data/us-historical-stock-prices-with-earnings-data/dataset_summary.csv",
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": ["symbol","total_prices","stock_from_date","stock_to_date","total_earnings","earnings_from_date","earnings_to_date"]
    },
)

document = loader.load()

# set prompt template
prompt_template = """
    Use the following pieces of context to answer the question at the end. 
    
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    You are an assistant that can help analyze the provided financial data in csv format.

    The provided data are a financial data that consists of 7 columns:
    * First column is symbol in which the company's ticker as string data type
    * Second column is the number of price data collected for a current stock ticker as integer data type
    * Third column is the data info of when stock's price is first collected as date data type
    * Fourth column is the data info of when stock's price is last being collected as date data type
    * Fifth column is the data that consists of number of total earnings data collected for the company in integer data type
    * Sixth column is the data info of when stock's earnings is first collected as date data type
    * Seventh column is the data info of when stock's earnings is last being collected as date data type
    
    {context}

    Please use financial knowledge to answer the existing question.

    Here are some examples of how you should answer the questions.

    Question: How many earnings for A ticker does the current csv data has?
    Answer: 49

    Question: How many prices are collected for A ticker does the current csv data has?
    Answer: 5518

    Question: How many stock tickers does the csv data has?
    Answer: 7786
    
    Question: What is the earliest price data does A ticker was being collected?
    Answer: 1999-11-18

    Question: What is the last price data does A ticker was being collected?
    Answer: 2021-06-02

    Question: What is the earliest earning data does A ticker was being collected?
    Answer: 2009-05-14

    Question: What is the last earning data does A ticker was being collected?
    Answer: 2021-05-25
    
    Question: {question}
    Answer:"""

prompt_template_1 = """
    [INST] 
    Respond to the prompt using the data from pandas dataframe with the given context

    {context}

    <<SYS>>
    You are a helpful, respectful and honest financial chat assistant. 
    
    Your primarily responsibility is to help analyze large csv data that contains financial information using Pandas DataFrame
    
    Here are some previous conversations between the Assistant and User:

    User: How many rows are there in the csv?
    Assistant: 7786 rows

    User: show top 10 rows of the data?
    Assistant: df.head(10)

    <</SYS>>
    {prompt}
    [/INST]
"""

prompt_template_2 = """
    Use the following pieces of context to answer the question at the end. 
    
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    You are an assistant that can help analyze the provided financial data in csv format.

    The provided data are a financial data that consists of 7 columns:
    * First column is symbol in which the company's ticker as string data type
    * Second column is the number of price data collected for a current stock ticker as integer data type
    * Third column is the data info of when stock's price is first collected as date data type
    * Fourth column is the data info of when stock's price is last being collected as date data type
    * Fifth column is the data that consists of number of total earnings data collected for the company in integer data type
    * Sixth column is the data info of when stock's earnings is first collected as date data type
    * Seventh column is the data info of when stock's earnings is last being collected as date data type
    
    {context}

    Please use financial knowledge to answer the existing question.

    Here are some examples of how you should answer the questions.

    Question: How many prices are collected for A ticker does the current csv data has?
    Answer: 5518

    Question: How many stock tickers does the csv data has?
    Answer: 7786
    
    Question: What is the earliest price data does A ticker was being collected?
    Answer: 1999-11-18

    Question: What is the last price data does A ticker was being collected?
    Answer: 2021-06-02
    
    Question: {question}
    Answer:"""



# prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

prompt = PromptTemplate(template=prompt_template_1, input_variables=["context", "prompt"])

text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_documents(document)

## initliaze chromadb
model_name = "BAAI/bge-small-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vectordb = Chroma.from_documents(texts,hf)

question = "Show top 5 rows of the data?"

# similar_doc = vectordb.similarity_search(question, k=1)
# context = similar_doc[0].page_content
context = vectordb.as_retriever()
query_llm = LLMChain(llm=llm, prompt=prompt)
response = query_llm.run({"context": context, "prompt": question})

print(response)
