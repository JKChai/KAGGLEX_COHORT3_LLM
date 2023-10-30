from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
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
    {question}
    [/INST]
"""

# prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

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

chain_type_kwargs = {"prompt": prompt}
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectordb.as_retriever(), chain_type_kwargs=chain_type_kwargs)

question = "Show top 5 rows of the data"

response = qa.run(question)

# similar_doc = vectordb.similarity_search(question, k=1)
# context = similar_doc[0].page_content
# context = vectordb.as_retriever()
# query_llm = LLMChain(llm=llm, prompt=prompt)
# response = query_llm.run({"context": context, "prompt": question})

print(response)
