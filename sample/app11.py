
from pandasai import PandasAI
from pandasai import SmartDataframe
from pandasai.llm import Falcon

import pandas as pd

## Hugging FACE API KEY
HUGGINGFACE_API_KEY = "API_KEY_HERE"

# Falcon
## Deprecated model
llm = Falcon(api_token=HUGGINGFACE_API_KEY)

df = SmartDataframe("./data/us-historical-stock-prices-with-earnings-data/dataset_summary.csv", config={"llm": llm})

# df = pd.read_csv("./data/us-historical-stock-prices-with-earnings-data/dataset_summary.csv")
# df.head(3)

# pandas_ai = PandasAI(llm)

# print(pandas_ai.run(df, prompt='How many rows are there?'))
query = "How many rows are there?"

## FAIL CHAT
print(df.chat(query))