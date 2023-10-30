
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import csv_loader

## generate llm model
llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGUF", model_file="llama-2-7b-chat.Q5_K_M.gguf", model_type="llama")

## load data
def _load_documents():
    loader1 = csv_loader.CSVLoader(
        file_path="./data/us-historical-stock-prices-with-earnings-data/dataset_summary.csv",
        csv_args={
            "delimiter": ",",
            "quotechar": '"',
            "fieldnames": ["symbol","total_prices","stock_from_date","stock_to_date","total_earnings","earnings_from_date","earnings_to_date"],
        },
    )

    loader2 = csv_loader.CSVLoader(
        file_path="./data/us-historical-stock-prices-with-earnings-data/stocks_latest/stock_prices_latest.csv",
        csv_args={
            "delimiter": ",",
            "quotechar": '"',
            "fieldnames": ["symbol","date","open","high","low","close","close_adjusted","volume","split_coefficient"],
        },
    )

    ## collect loaders
    loaders = [loader1, loader2]

    ## create document 
    docs = []

    for loader in loaders:
        docs.extend(loader.load())

    return docs

## Split Documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(_load_documents())

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

## create memory object
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

## Create Chain
qa = ConversationalRetrievalChain.from_llm(llm=llm, vectorstore=vectordb.as_retriever(), memory=memory)

# qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=vectordb)

## Ask Questions
query = "when does 'MSFT' ticker has the highest price?"

result = qa({"question": query})

print(result["answer"])