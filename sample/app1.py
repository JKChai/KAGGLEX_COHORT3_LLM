import os
import streamlit as st
import pandas as pd
from langchain_experimental.agents import agent_toolkits
from langchain.llms import CTransformers

# Define Streamlit app
def app():
      # Title and description
    st.title("CSV Query App")
    st.write("Upload a CSV file and enter a query to get an answer.")
    file =  st.file_uploader("Upload CSV file",type=["csv"])

    if file is not None:
        data = pd.read_csv(file)
        st.write("Data Preview:")
        st.dataframe(data.head()) 

    else:
        st.write("File has None Type FORCE STOP")
        st.stop()

    # llm = CTransformers(model="TheBloke/Llama-2-13B-chat-GGUF", model_file="llama-2-13b-chat.Q5_K_M.gguf", model_type="llama")

    # agent = create_pandas_dataframe_agent(OpenAI(temperature=0),data,verbose=True) 

    query = st.text_input("Enter a query:") 

    if st.button("Execute"):
        # answer = agent.run(query)
        st.write("Answer:")
        # st.write(answer)    
  
if __name__ == "__main__":
    app()   