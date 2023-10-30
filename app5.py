
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
# from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
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

    # loader2 = csv_loader.CSVLoader(
    #     file_path="./data/portfolio/SamplePortfolio.csv",
    #     csv_args={
    #         "delimiter": ",",
    #         "fieldnames": ["UNIQUE ID","ID TYPE","NAME","POSITION UNITS","PRICE","CURRENCY","TICKER","Asset Class","sector","geography"],
    #     },
    # )

    # ## collect loaders
    # loaders = [loader1, loader2]

    # ## create document 
    # docs = []

    # for loader in loaders:
    #     docs.extend(loader.load())
    docs = loader1.load()

    return docs

## Generate Prompt template


prompt_template = """
    Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    You are a data assistant that can help analyze csv data that consists of financial information.
    
    {context}

    Question: {question}
    Answer the questions as a financial analyst:
    """

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

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

# ## create memory object
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

## Create Chain
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=vectordb)

## Ask Questions
query = "What stock ticker has the best return?"

result = qa.run(query)

print(result)