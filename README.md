# SMCDC2024
AI-Assisted Tool for Climate Research 

Climate change is a major threat to international society.  Climate change causes more and more disasters, including intense droughts, water scarcity, severe fires, rising sea levels, flooding, melting polar ice, and many others. It is critical and urgent for the government to adopt correct climate policy and for nationals to collaborate. However, government policies are often determined by a small group of governors who do not always put climate change as the top priority. The general public is not involved much in climate policy decisions. One reason is the gap between cutting-edge climate research results and the poor efficacy in disseminating them to the general public. Another reason is the general public often cannot find the time or lack of climate knowledge to read and understand the long climate report provided by the government. Another reason is that the US is an immigrant country. There is a large non-English-speaking population in this country. It is even harder for people who speak foreign languages to be involved with climate policies.

Motivated by the above challenges, this work aims to develop an AI toolbox that can read and digest scientific climate papers and government climate reports, and then generate frequently asked questions (FAQs), questions & answers (Q&As), and main idea summaries for the peruse of the general multilingual public.   

Python code is developed for this purpose.

Install required libraries:
- pip install transformers torch
  
Workflow:
- load the Llama-2-7b-chat-hf model and tokenizer.
- Read the input file and generate Q&A pairs and Summaries. 
- Save and display the output.
- Translate the output into other languages.
- Convert output into audio mp3 files.
  
Note:

Generate Q&A pairs : Create_Q_A.py
Generate Summaries:  Create_Summary.py
Translate into other languages: translate.py
Convert output into audio mp3 file: voice.py
