## Try to build a CSV chatbot with pandasai using OSS LLM
## pandas ai doesn't work well with this llm model
import pandas as pd

from pandasai import SmartDataframe
from langchain.llms import CTransformers

llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGUF", model_file="llama-2-7b-chat.Q5_K_M.gguf", model_type="llama")

df = SmartDataframe("./data/us-historical-stock-prices-with-earnings-data/dataset_summary.csv", config={"llm": llm})

print(df.chat('How many rows are there in the dataset?'))

prompt = """
    [INST] 
    <<SYS>>
    You are a helpful, respectful and honest financial chat assistant. 
    
    Your primarily responsibility is to help analyze large csv data that contains financial information.
    
    Always answer as helpfully as possible, while being safe. 
    Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
    Please ensure that your responses are socially unbiased and positive in nature. 
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
    If you don't know the answer to a question, please don't share false information.

    <</SYS>>
    {prompt}
    [/INST]
"""
