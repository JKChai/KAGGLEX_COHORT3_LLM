from langchain.llms import CTransformers
from langchain.agents.agent_types import AgentType

from langchain_experimental.agents.agent_toolkits import create_csv_agent

llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGUF", model_file="llama-2-7b-chat.Q5_K_M.gguf", model_type="llama", temperature=0)

agent = create_csv_agent(
    llm,
    "./data/us-historical-stock-prices-with-earnings-data/dataset_summary.csv",
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

agent.run("how many rows are there?")