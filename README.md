# ----NewsBit: News Summarization and Fact-Checking App---

NewsBit is a web application that summarizes news articles from URLs, answers questions about their content, and fact-checks claims using Wikipedia. Built with LangChain, Streamlit, and the Hugging Face Inference API, it leverages retrieval-augmented generation (RAG) for question-answering and an agent-based approach for fact-checking.


# ---Problem Statement
The rapid growth of online news has made it challenging for users to quickly grasp key information from lengthy articles while ensuring the accuracy of the content. Many news articles contain unverified claims, and manually fact-checking them is time-consuming. This project aims to address these issues by developing a web application that:

# ---Summarizes news articles from a given URL into concise 50-100 word overviews.
Allows users to ask questions about the article for deeper understanding.
Fact-checks claims within the article using Wikipedia to ensure credibility.The application provides an interactive and user-friendly interface to streamline news consumption and promote informed decision-making.

# ---Explanation of Agent Interactions
The application leverages multiple agents and components to process, summarize, query, and fact-check news articles. Below is a breakdown of the interactions:

# ---Article Fetching Agent:

The system uses the newspaper3k library to extract text from a user-provided URL. If extraction fails (e.g., due to paywalls or unsupported formats), it falls back to LangChain’s WebBaseLoader to scrape the content.
The fetched text is converted into a LangChain Document object for further processing.


# ---Text Processing and Embedding Agent:

The article text is split into manageable chunks (1000 characters with 100-character overlap) using LangChain’s RecursiveCharacterTextSplitter.
Chunks are embedded into vectors using HuggingFaceEmbeddings (model: sentence-transformers/all-MiniLM-L12-v2) and stored in a FAISS vector store for efficient retrieval.


# ---Summarization Agent:

A map_reduce summarization chain, powered by Hugging Face’s Mixtral-8x7B-Instruct-v0.1 model, processes the text chunks.
Each chunk is summarized into 50 words or less, and the summaries are combined into a 50-100 word overview using custom prompts.


# ---Question Answering Agent:

A ConversationalRetrievalChain retrieves the top 3 relevant chunks from the FAISS vector store based on the user’s question.
The Mixtral-8x7B model generates an answer, incorporating chat history for context-aware responses.
Source document excerpts are displayed to provide transparency.


# ---Fact-Checking Agent:

A LangChain agent with a Wikipedia-based tool is triggered for queries containing phrases like “fact-check” or “is this true.”
The agent queries the wikipediaapi library to fetch relevant information and returns a summary of the findings (first 200 characters of the Wikipedia page summary).


# ---Caching Agent:

The article content is hashed using MD5, and the FAISS vector store is saved locally to avoid redundant processing.
On subsequent requests for the same article, the cached vector store is loaded to improve performance.


# ---User Interface Agent (Streamlit):

Streamlit manages the user interface, handling inputs (URL, questions), displaying outputs (summary, answers, source documents), and maintaining state (chat history, vector store).
It provides visual feedback (e.g., spinners, error messages) to enhance user experience.



These agents interact seamlessly to fetch, process, summarize, and analyze news articles while ensuring efficient performance through caching and a robust UI.
Technologies Used

Python 3.8+: Core programming language for the application.
Streamlit: Framework for building the interactive web interface.
LangChain: For text splitting (RecursiveCharacterTextSplitter), embeddings (HuggingFaceEmbeddings), vector storage (FAISS), summarization (load_summarize_chain), and question answering (ConversationalRetrievalChain).
HuggingFaceEndpoint: For NLP tasks using the Mixtral-8x7B-Instruct-v0.1 model.
FAISS: For efficient similarity search in embedded text chunks.
newspaper3k: For extracting article content from URLs.
WebBaseLoader: Fallback for article extraction when newspaper3k fails.
wikipediaapi: For fact-checking claims using Wikipedia.
HuggingFaceEmbeddings: For generating text embeddings (model: sentence-transformers/all-MiniLM-L12-v2).
dotenv: For managing environment variables (e.g., Hugging Face API token).
hashlib: For generating MD5 hashes for caching.
pathlib: For handling file paths in the caching mechanism.

# ---Setup and Run Instructions---
Follow these steps to set up and run the News Summarizer and Fact-Checker application locally:

Clone the Repository:
git clone [https://github.com/mohdsami01/NewsBits.git]
cd [URL_NewsBits]


Set Up a Virtual Environment (Optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:

Ensure you have Python 3.8 or higher installed.
Install the required packages using the provided requirements.txt (you’ll need to create this file based on the dependencies listed below):pip install streamlit langchain langchain-huggingface langchain-community faiss-cpu newspaper3k wikipedia-api python-dotenv




Set Up Environment Variables:

Create a .env file in the project root directory.
Add your Hugging Face API token:HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token


You can obtain the token from your Hugging Face account (https://huggingface.co/settings/tokens).


Run the Application:

Start the Streamlit app:streamlit run app.py


Open your browser and navigate to http://localhost:8501 (or the URL provided by Streamlit).


Usage:

Enter a news article URL in the input field (e.g., https://www.bbc.com/news/article).
View the generated summary.
Ask questions about the article or fact-check claims (e.g., “Is this true?”).
Review the chat history and source documents for transparency.


Troubleshooting:

Ensure your internet connection is active, as the app relies on Hugging Face’s API.
If the app fails to fetch an article, try a different URL or check for paywalls.
Verify that the .env file contains the correct API token.



GitHub Repository Link

GitHub Repository: [https://github.com/mohdsami01/NewsBits]

Deployment/Demo Link: [https://newsbits-6q0z.onrender.com/]



# ***----- DEMO QUE TO ASK-------***


1-> What are the main topics covered on the BBC News homepage today?
2-> What is the top story on the BBC News website?
3-> Are there any technology-related news stories mentioned?
4-> What does the BBC News page say about climate change today?

# **-- FACT CHECK QUESTION----**

1-> Fact-check: Is climate change primarily caused by human activity?
2-> Fact-check: Is the United Nations involved in global peacekeeping?

# ----***DEMO LINK TO PASTE ON THE STREAMLIT APPLICATION BEFORE ASKING QUE****----
// [https://www.bbc.com]