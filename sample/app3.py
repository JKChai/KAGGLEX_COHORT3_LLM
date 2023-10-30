import pandas as pd

from langchain_experimental.agents import agent_toolkits
from langchain.document_loaders import DataFrameLoader, csv_loader
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import CTransformers

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


"""
    Always answer as helpfully as possible, while being safe. 
    Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
    Please ensure that your responses are socially unbiased and positive in nature. 
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
    If you don't know the answer to a question, please don't share false information.
"""

prompt_template = """
    [INST] 
    Respond to the prompt using the data from pandas dataframe

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

prompt_question = "Show top 5 rows of the data?"

llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGUF", model_file="llama-2-7b-chat.Q5_K_M.gguf", model_type="llama")

chat = llm()

print(chat[HumanMessage(content="Translate this sentence from English to French: I love programming.")])

# df = pd.read_csv('./data/us-historical-stock-prices-with-earnings-data/dataset_summary.csv')

# loader = DataFrameLoader(df, page_content_column="symbol")

# prompt = PromptTemplate(template=prompt_template, input_variables=["prompt"])

# llm_chain = LLMChain(llm=llm, prompt=prompt,)

# response = llm_chain.run({"prompt": prompt_question})     

# print(response)

# agent = agent_toolkits.create_pandas_dataframe_agent(llm=llm,df=df,verbose=True) 

# print(agent.run("Count number of rows"))

loader = csv_loader.CSVLoader(
    file_path="./data/us-historical-stock-prices-with-earnings-data/dataset_summary.csv",
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": ["symbol","total_prices","stock_from_date","stock_to_date","total_earnings","earnings_from_date","earnings_to_date"],
    },
)

data = loader.load()
