<!-- Template Adapted & Provided by https://github.com/othneildrew/Best-README-Template -->
<a name="readme-top"></a>

<br />
<div align="center">
  <h1 align="center">KAGGLEX COHORT3 - Build My Own ChatBot Assistant</h1>
  <p align="center">
    Project built & presented for KAGGLEX COHORT 3 PROGRAM
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<h2>Table of Contents</h2>
<ol>
    <li>
        <a href="#readme">README</a>
        <ul>
            <li><a href="#prerequisites">Prerequisites</a></li>        
        </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    <li><a href="#references">References</a></li>
</ol>

## README

This is a learning project built for KaggleX - Cohort 3 program. The goal for this project is to learn to build my own Kaggle project, in this case, a "chatbot". Although I was not able to completely run it on Kaggle Notebook, however, Kaggle has been one of the important resources for me to research & learn & collect the data I need for this project.

This "chatbot" should be able to run on personal pc as it utilizes Quantization technique & uses LLAMA2, an open-source model, without requiring extra dollars to run the "chatbot".

Users who has less than 36GB RAM should run the model with 7b-llama model as 13-b model can consume the RAM significantly using chat history process.

Note that the "chatbot" is still limited and has not able to complete the tasks that I am expecting it to complete such as providing financial information based on the given financial data, providing financial advice based on the user portfolio, etc.

Any developers can try to improve & extend this project by:

* Putting more effort on Prompt Engineering
* Training the model to learn on a domain only task for precisions
* Help make `create_pandas_dataframe_agent` & `csvloader` better by contributing to the source code

### Prerequisites

* [![LLAMA2][llama2-shield]][llama2-url]
* [![LangChain][langchain-shield]][langchain-url]
* [![Streamlit][streamLit-shield]][streamlit-url] 

Note that I uses [`pipenv`](https://pypi.org/project/pipenv/) for Package Management instead of `pip`. 

To install [pipenv](https://docs.pipenv.org/basics/), run the below command.

```pwsh
C:\Users\USERNAME> Python -m pip install pipenv
``` 

After `pipenv` is installed, run the command below from the repo directory.
This command will install all packages listed in `Pipfile`.
Note that all required packages are listed in `Pipfile` & required versions for these packages are listed in `Pipfile.lock`

```pwsh
C:\Users\USERNAME\repo\KAGGLEX3_LLM> pipenv install
``` 

You can check if `pipenv` is installed by running this command.

```pwsh
C:\Users\USERNAME\repo\KAGGLEX3_LLM> pip list pipenv
``` 

If you decided not to use `pipenv`, you might have to install all packages manually using `pip`.

## Usage

To run the script, run the below command.

```pwsh
C:\Users\USERNAME\repo\KAGGLEX3_LLM> pipenv run streamlit run .\app.py
```

Note that ```python llm()``` function created in `app.py` required an integer `type` parameter input, this input is listed as below.

```python
{
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
```

Each model will required a package download to `C:\Users\USERNAME\.cache\huggingface` & the size of each model is listed as below.

| NAME | Quant Method | Bits | Size | MAX RAM REQUIRED |
| --- | --- | --- | --- | --- |
| llama-2-7b-chat.Q4_K_M.gguf	 | Q4_K_M | 4 | 4.08 GB  | 6.58 GB |
| llama-2-7b-chat.Q5_K_M.gguf  | Q5_K_M | 5 | 4.78 GB  | 7.28 GB |
| llama-2-7b-chat.Q6_K.gguf    | Q6_K   | 6 | 5.53 GB  | 8.03 GB |
| llama-2-7b-chat.Q8_0.gguf    | Q8_0   | 8 | 7.16 GB  | 9.66 GB |
| llama-2-13b-chat.Q4_K_M.gguf | Q4_K_M | 4 | 7.87 GB  | 10.37 GB |
| llama-2-13b-chat.Q5_K_M.gguf | Q5_K_M | 5 | 9.23 GB  | 11.47 GB |
| llama-2-13b-chat.Q6_K.gguf   | Q6_K   | 6 | 10.68 GB | 13.18 GB |
| llama-2-13b-chat.Q8_0.gguf   | Q8_0   | 8 | 13.83 GB | 16.33 GB |




## Roadmap

- [x] Define Project Goal & Problem Statement
- [x] Learn about Prompt Engineering & LangChain
- [x] Build a Prototype of the LLM APP
- [x] Test & Run different examples to try finding the best way of building this Chat
- [x] Test & Run different loaders to work with input data for the Chat
- [x] Test & Run different LLM models with LangChain that support Quantization
- [x] Test the built Chat Application
- [x] Refine the Chat Application for better accuracy 
- [x] Organize documentation & all works for PPT presentation
- [x] Complete Markdown
- [ ] Upload scripts & docs to GitHub & make it Public
- [ ] Submit PPT presentation & 5 Mins video

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

I would like to thank all of KaggleX's staff & presenters from `Cohort 3 of the KaggleX BIPOC Mentorship Program` for taking your time & effort to make this happen :blush:. I would also like to thank my Mentor `Emmanuel Mayssat` for sharing project ideas & advice providing me a good headstart to explore & research about Prompt Engineering building my own Chat Assistant that can run on my CPU machine :grin:. Thank you to KaggleX Discord Community as well for sharing ideas & advice. Because of all of you, these 3 months journey have been challenging but fruitful providing me a different opportunity to learn & explore new & different ways of manipulating & handling data. :joy:

Thank you all & all the best! :satisfied:

## References

Multiple resources have been used for testing, experimenting, learning, & researching.

Tool's documentation is listed below:

* https://python.langchain.com
* https://smith.langchain.com
* https://docs.streamlit.io/
* https://docs.trychroma.com
* https://faiss.ai/index.html
* https://docs.pandas-ai.com/en/latest/
* https://huggingface.co/sentence-transformers
* https://pypi.org/project/InstructorEmbedding/

Some important references on `langchain` & `langchain_experimental` modules:

* https://api.python.langchain.com/en/latest/experimental_api_reference.html
* https://api.python.langchain.com/en/latest/_modules/langchain_experimental/agents/agent_toolkits/pandas/base.html
* https://python.langchain.com/docs/integrations/text_embedding/bge_huggingface
* https://python.langchain.com/docs/integrations/document_loaders/pandas_dataframe
* https://python.langchain.com/docs/use_cases/question_answering/

Hugging Face model references:

* https://huggingface.co/FinGPT
* https://huggingface.co/tiiuae/falcon-7b
* https://huggingface.co/TheBloke/Llama-2-7B-chat-GGUF
* https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF

Kaggle references:

* https://www.kaggle.com/datasets/tsaustin/us-historical-stock-prices-with-earnings-data
* https://www.kaggle.com/code/gpreda/rag-using-llama-2-langchain-and-chromadb


Tool's references from GitHub are listed below:
 
* https://github.com/ggerganov/llama.cpp
* https://github.com/gventuri/pandas-ai
* https://github.com/AI4Finance-Foundation/FinNLP
* https://github.com/AI4Finance-Foundation/FinGPT
* https://github.com/mrspiggot/LucidateFinAgent
* https://github.com/afaqueumer/DocQA

Code's references from GitHub are listed below:

* https://github.com/hwchase17/chat-your-data
* https://github.com/hwchase17/chroma-langchain/blob/master/qa.ipynb
* https://github.com/InsightEdge01/Chat-CSV/blob/main/appfile.py
* https://github.com/AIAnytime/ChatCSV-Llama2-Chatbot
* https://github.com/AIAnytime/ChatCSV-Streamlit-App
* https://github.com/pinecone-io/examples/blob/master/learn/generation/llm-field-guide/llama-2/llama-2-70b-chat-agent.ipynb
* https://github.com/langchain-ai/langchain-benchmarks/blob/main/csv-qa/streamlit_app.py
* https://github.com/IBM/Analyze-Investment-Portfolio/blob/master/SamplePortfolio.csv
* https://github.com/sudarshan-koirala/youtube-stuffs/blob/main/langchain/Chat_with_Any_Documents_Own_ChatGPT_with_LangChain.ipynb

Other references:

* https://www.kdnuggets.com/build-your-own-pandasai-with-llamaindex
* https://www.datacamp.com/blog/an-introduction-to-pandas-ai
* https://dev.to/ngonidzashe/chat-with-your-csv-visualize-your-data-with-langchain-and-streamlit-ej7
* https://realpython.com/practical-prompt-engineering/
* https://github.com/othneildrew/Best-README-Template
* https://gist.github.com/rxaviers/7360908

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[streamlit-shield]: https://img.shields.io/badge/Streamlit-000000?style=for-the-badge&logo=Streamlit
[streamlit-url]: https://streamlit.io/
[llama2-shield]: https://img.shields.io/badge/LLAMA2-blue
[llama2-url]: https://huggingface.co/docs/transformers/main/model_doc/llama2
[langchain-shield]: https://img.shields.io/badge/%F0%9F%A6%9C%EF%B8%8F%F0%9F%94%97-LANGCHAIN-000000
[langchain-url]: https://www.langchain.com/
