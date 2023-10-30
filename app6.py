## Keep failing as pandas dataframe agent has lots of limitation
import pandas as pd

from langchain.llms import CTransformers
from langchain_experimental.agents import agent_toolkits

llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGUF", model_file="llama-2-7b-chat.Q5_K_M.gguf", model_type="llama")

df = pd.read_csv("./data/us-historical-stock-prices-with-earnings-data/dataset_summary.csv")

agent = agent_toolkits.create_pandas_dataframe_agent(llm=llm,df=df) 

query = "How many rows are there?"

agent.run(query)
