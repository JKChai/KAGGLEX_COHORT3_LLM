from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import Chroma

llm = CTransformers(model="TheBloke/Llama-2-13B-Ensemble-v5-GGUF", model_file="llama-2-13b-ensemble-v5.Q5_K_M.gguf", model_type="llama")

template = """
              Write a concise summary of the following text delimited by triple backquotes.
              Return your response in bullet points which covers the key points of the text.
              ```{text}```
              BULLET POINT SUMMARY:
           """

prompt = PromptTemplate(template=template, input_variables=["text"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

text1 = """
Tesla, Inc. (/ˈtɛslə/ TESS-lə or /ˈtɛzlə/ TEZ-lə[a]) is an American multinational automotive and clean energy company headquartered in Austin, Texas. Tesla designs and manufactures electric vehicles (cars and trucks), stationary battery energy storage devices from home to grid-scale, solar panels and solar roof tiles, and related products and services.

Tesla is one of the world's most valuable companies and, as of 2023, was the world's most valuable automaker. In 2022, the company led the battery electric vehicle market, with 18% share.

Its subsidiary Tesla Energy develops and is a major installer of photovoltaic systems in the United States. Tesla Energy is one of the largest global suppliers of battery energy storage systems, with 6.5 gigawatt-hours (GWh) installed in 2022.

Tesla was incorporated in July 2003 by Martin Eberhard and Marc Tarpenning as Tesla Motors. The company's name is a tribute to inventor and electrical engineer Nikola Tesla. In February 2004, via a $6.5 million investment, Elon Musk became the company's largest shareholder. He became CEO in 2008. Tesla's announced mission is to help expedite the move to sustainable transport and energy, obtained through electric vehicles and solar power.

Tesla began production of its first car model, the Roadster sports car, in 2008. This was followed by the Model S sedan in 2012, the Model X SUV in 2015, the Model 3 sedan in 2017, the Model Y crossover in 2020, and the Tesla Semi truck in 2022. The company plans production of the Cybertruck light-duty pickup truck in 2023.[8] The Model 3 is the all-time bestselling plug-in electric car worldwide, and in June 2021 became the first electric car to sell 1 million units globally.[9] Tesla's 2022 deliveries were around 1.31 million vehicles, a 40% increase over the previous year,[10][11] and cumulative sales totaled 4 million cars as of April 2023.[12] In October 2021, Tesla's market capitalization temporarily reached $1 trillion, the sixth company to do so in U.S. history.

Tesla has been the subject of lawsuits, government scrutiny, and journalistic criticism, stemming from allegations of whistleblower retaliation, worker rights violations, product defects, and Musk's many controversial statements.
"""

print(llm_chain.run(text1))

text2 ="""
Apple Inc. is an American multinational technology company headquartered in Cupertino, California. Apple is the world's largest technology company by revenue, with US$394.3 billion in 2022 revenue.[6] As of March 2023, Apple is the world's biggest company by market capitalization.[7] As of June 2022, Apple is the fourth-largest personal computer vendor by unit sales and the second-largest mobile phone manufacturer in the world. It is often considered as one of the Big Five American information technology companies, alongside Alphabet (parent company of Google), Amazon, Meta Platforms, and Microsoft.

Apple was founded as Apple Computer Company on April 1, 1976, by Steve Wozniak, Steve Jobs (1955–2011) and Ronald Wayne to develop and sell Wozniak's Apple I personal computer. It was incorporated by Jobs and Wozniak as Apple Computer, Inc. in 1977. The company's second computer, the Apple II, became a best seller and one of the first mass-produced microcomputers. Apple went public in 1980 to instant financial success. The company developed computers featuring innovative graphical user interfaces, including the 1984 original Macintosh, announced that year in a critically acclaimed advertisement called "1984". By 1985, the high cost of its products, and power struggles between executives, caused problems. Wozniak stepped back from Apple and pursued other ventures, while Jobs resigned and founded NeXT, taking some Apple employees with him.

As the market for personal computers expanded and evolved throughout the 1990s, Apple lost considerable market share to the lower-priced duopoly of the Microsoft Windows operating system on Intel-powered PC clones (also known as "Wintel"). In 1997, weeks away from bankruptcy, the company bought NeXT to resolve Apple's unsuccessful operating system strategy and entice Jobs back to the company. Over the next decade, Jobs guided Apple back to profitability through a number of tactics including introducing the iMac, iPod, iPhone and iPad to critical acclaim, launching the "Think different" campaign and other memorable advertising campaigns, opening the Apple Store retail chain, and acquiring numerous companies to broaden the company's product portfolio. When Jobs resigned in 2011 for health reasons, and died two months later, he was succeeded as CEO by Tim Cook.

Apple became the first publicly traded U.S. company to be valued at over $1 trillion in August 2018, then at $2 trillion in August 2020, and at $3 trillion in January 2022. As of April 2023, it was valued at around $2.6 trillion. The company receives criticism regarding the labor practices of its contractors, its environmental practices, and its business ethics, including anti-competitive practices and materials sourcing. Nevertheless, the company has a large following and enjoys a high level of brand loyalty. It has also been consistently ranked as one of the world's most valuable brands.
"""
print(llm_chain.run(text2))