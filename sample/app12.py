## Adopted from https://github.com/AIAnytime/ChatCSV-Llama2-Chatbot/blob/main/app.py
import tempfile
import streamlit as st 

from streamlit_chat import message
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain


#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGUF", 
        model_file="llama-2-7b-chat.Q8_0.gguf", 
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.0
    )
    return llm

def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

_file_path = "./data/us-historical-stock-prices-with-earnings-data/dataset_summary.csv"
_vsdb_path = './vectorstore/db_faiss'

st.title("## Simple RoboAdvisor for DEMO ##")
# st.markdown("<h3 style='text-align: center; color: white;'>Built by <a href='https://github.com/AIAnytime'>AI Anytime with ❤️ </a></h3>", unsafe_allow_html=True)

_import_file = st.sidebar.file_uploader("Upload your Data", type="csv")

if _import_file:
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

    ## Load Embedding Model
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs
            )

    db = FAISS.from_documents(data, embeddings)
    db.save_local(_vsdb_path)

    llm = load_llm()
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello User! Try asking me anything about " + _import_file.name]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hello ChatBot!"]

    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()

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


    # # result = chain({"question": query, "chat_history": st.session_state['history']})

    # query = "How many rows are there in this dataset?"

    # _chat_history = []

    # result = chain({"question": query, "chat_history": _chat_history})

    # print(result["answer"])