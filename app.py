## Adopted from https://github.com/AIAnytime/ChatCSV-Llama2-Chatbot/blob/main/app.py
import tempfile
import streamlit as st 

from streamlit_chat import message
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain

#Loading the model
def load_llm(type=0, temperature = 0.0, max_new_tokens=512):
    # Load the locally downloaded model here

    model_dict = {
        0:{
            "model":"TheBloke/Llama-2-7B-Chat-GGUF", 
            "model_file":"llama-2-7b-chat.Q4_K_M.gguf"
        },
        1:{
            "model":"TheBloke/Llama-2-7B-Chat-GGUF", 
            "model_file":"llama-2-7b-chat.Q5_K_M.gguf"
        },
        2:{
            "model":"TheBloke/Llama-2-7B-Chat-GGUF", 
            "model_file":"llama-2-7b-chat.Q6_K.gguf"
        },
        3:{
            "model":"TheBloke/Llama-2-7B-Chat-GGUF", 
            "model_file":"llama-2-7b-chat.Q8_0.gguf"
        },
        4:{
            "model":"TheBloke/Llama-2-13B-Chat-GGUF", 
            "model_file":"llama-2-13b-chat.Q4_K_M.gguf"
        },
        5:{
            "model":"TheBloke/Llama-2-13B-Chat-GGUF", 
            "model_file":"llama-2-13b-chat.Q5_K_M.gguf"
        },
        6:{
            "model":"TheBloke/Llama-2-13B-Chat-GGUF", 
            "model_file":"llama-2-13b-chat.Q6_K.gguf"
        },
        7:{
            "model":"TheBloke/Llama-2-13B-Chat-GGUF", 
            "model_file":"llama-2-13b-chat.Q8_0.gguf"
        }
    }

    llm = CTransformers(
        model=model_dict[type]["model"], 
        model_file=model_dict[type]["model_file"], 
        model_type="llama",
        max_new_tokens = max_new_tokens,
        temperature = temperature
    )

    return llm

# loading embdedded model
def embedding_model(type=0):

    model_kwargs = {'device': 'cpu'}

    if type == 0:
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs
                )

    elif type == 1:
        model_name = "BAAI/bge-small-en"
        encode_kwargs = {'normalize_embeddings': True}
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    elif type == 2:
        model_name = "hkunlp/instructor-xl"
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=model_name, 
            model_kwargs=model_kwargs
            )

    return embeddings

def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))

    return result["answer"]

_file_path = "./data/us-historical-stock-prices-with-earnings-data/dataset_summary.csv"
_vsdb_path = './vectorstore/db_faiss'

## Page Configuration
st.set_page_config(
    page_title="Simple RoboAdvisor for DEMO",
    page_icon=":robot:"
)

st.header("🐦 KaggleX DEMO - RoboAdvisor 📊")
st.write("Please upload a csv file to the left to kick on this ChatBot...")
_import_file = st.sidebar.file_uploader("Upload your Data", type="csv")

#container for the chat history
response_container = st.container()
#container for the user's text input
container = st.container()

if _import_file:

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello User! Try asking me anything about " + _import_file.name]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hello ChatBot!"]

   #use tempfile because CSVLoader only accepts a file_path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(_import_file.getvalue())
        tmp_file_path = tmp_file.name

    ## CSV LOADER
    loader = CSVLoader(
        file_path =_file_path, 
        encoding  ="utf-8", 
        csv_args={'delimiter': ','}
        )

    data = loader.load()

    ## Load Embedding Model to VectorDB
    db = FAISS.from_documents(data, embedding_model(type=0))
    db.save_local(_vsdb_path)

    ## Load model & initiate Chain
    llm = load_llm(type=1)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            
            user_input = st.text_input("Query:", placeholder="Talk to your csv data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

else:
    st.write("Not CSV file WAS FOUND!!")
