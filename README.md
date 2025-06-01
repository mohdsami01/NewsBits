# ----NewsBit: News Summarization and Fact-Checking App---

NewsBit is a web application that summarizes news articles from URLs, answers questions about their content, and fact-checks claims using Wikipedia. Built with LangChain, Streamlit, and the Hugging Face Inference API, it leverages retrieval-augmented generation (RAG) for question-answering and an agent-based approach for fact-checking.

# -----Features :--------------

1--Summarization: Generates 50–100 word summaries of news articles.

2--Question-Answering: Answers user questions based on article content using RAG.

3--Fact-Checking: Verifies claims via Wikipedia (with optional SerpAPI integration).

4--Caching: Stores processed articles in FAISS for faster reloads.

5--User Interface: Streamlit-based UI for easy interaction.

# ---Tech Stack:--

Framework: LangChain (0.3.6), Streamlit (1.39.0)

LLM: Hugging Face Inference API (mistralai/Mixtral-8x7B-Instruct-v0.1)

Embeddings: sentence-transformers/all-MiniLM-L12-v2

Vector Store: FAISS (1.8.0)

Article Parsing: newspaper3k (0.2.8), lxml_html_clean (0.2.2)

Fact-Checking: wikipedia-api (0.7.1)

# ---Prerequisites :--

Python: 3.10 or higher

Hugging Face API Token: Obtain from Hugging Face

Windows: Commands are tailored for Windows (tested on PowerShell)

Internet Connection: Required for API calls and article fetching


# -----Starting Steps--------

1----Navigate to Project Directory----

cd C:\Users\mohds\Desktop\URL_NewsBits

2----Set Up a Virtual Environment-----

python -m venv venv

3----Install the Required Packages-----

pip install -r requirements.txt

4-----Add Your Hugging Face API Token---

HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

5----Run the App-----

streamlit run app.py
