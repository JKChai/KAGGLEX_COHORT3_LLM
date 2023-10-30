from time import time

from langchain.llms import CTransformers
from langchain.document_loaders import csv_loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough

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

template = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    
    Question: {question} 
    Context: {context} 
    Answer:
    """

rag_prompt_custom = PromptTemplate.from_template(template)

document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
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

retriever = vectordb.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | rag_prompt_custom 
    | llm 
)

def test_rag(qa, query):
    print(f"Query: {query}\n")
    time_1 = time()
    result = qa.run(query)
    time_2 = time()
    print(f"Inference time: {round(time_2-time_1, 3)} sec.")
    print("\nResult: ", result)

def invoke_rag(qa, query):
    print(f"Query: {query}\n")
    time_1 = time()
    result = rag_chain.invoke(query)
    time_2 = time()
    print(f"Inference time: {round(time_2-time_1, 3)} sec.")
    print("\nResult: ", result)

query = "How many rows are there in the given data?"
invoke_rag(qa, query)
