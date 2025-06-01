import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
import os
from dotenv import load_dotenv
from pathlib import Path
import hashlib
import wikipediaapi
from newspaper import Article
from langchain_core.documents import Document

load_dotenv()
hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_api_token:
    st.error(
        "Hugging Face API token not found. Please set HUGGINGFACEHUB_API_TOKEN in .env."
    )
    st.stop()

st.set_page_config(page_title="News Summarizer & Fact-Checker", layout="wide")
st.title("📰 News Summarizer & Fact-Checker")
st.write(
    "Enter a news article URL to summarize it, ask questions, or fact-check claims!"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "article_hash" not in st.session_state:
    st.session_state.article_hash = None
if "summary" not in st.session_state:
    st.session_state.summary = None


def get_article_hash(content):
    return hashlib.md5(content.encode()).hexdigest()


def fetch_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        if not text.strip():

            loader = WebBaseLoader(url)
            docs = loader.load()
            text = docs[0].page_content
        if not text.strip():
            raise ValueError("No text extracted from the article.")
        return text
    except Exception as e:
        st.error(f"Error fetching article: {str(e)}")
        return None


def process_article(url):
    try:

        text = fetch_article(url)
        if not text:
            return None, None, None

        doc = Document(page_content=text, metadata={"source": url})

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        split_docs = text_splitter.split_documents([doc])

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L12-v2",
            encode_kwargs={"batch_size": 32},
        )
        vectorstore = FAISS.from_documents(documents=split_docs, embedding=embeddings)

        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            huggingfacehub_api_token=hf_api_token,
            temperature=0.7,
            max_length=512,
        )
        summary_chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=PromptTemplate.from_template(
                "Summarize this in 50 words or less: {text}"
            ),
            combine_prompt=PromptTemplate.from_template(
                "Combine these summaries into a concise overview (50-100 words): {text}"
            ),
        )
        summary = summary_chain.run(split_docs)

        cache_dir = Path("faiss_cache")
        cache_dir.mkdir(exist_ok=True)
        article_hash = get_article_hash(text)
        vectorstore.save_local(cache_dir / article_hash)

        return vectorstore, article_hash, summary
    except Exception as e:
        st.error(f"Error processing article: {str(e)}")
        return None, None, None


def load_cached_vectorstore(article_hash):
    try:
        cache_dir = Path("faiss_cache")
        if (cache_dir / article_hash).exists():
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L12-v2",
                encode_kwargs={"batch_size": 32},
            )
            return FAISS.load_local(
                cache_dir / article_hash,
                embeddings,
                allow_dangerous_deserialization=True,
            )
        return None
    except Exception as e:
        st.warning(f"Error loading cached vector store: {str(e)}")
        return None


def initialize_qa_chain(vectorstore):
    try:
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            huggingfacehub_api_token=hf_api_token,
            temperature=0.7,
            max_length=512,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=retriever, return_source_documents=True
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error initializing QA chain: {str(e)}")
        return None


def wikipedia_fact_check(query):
    try:
        wiki = wikipediaapi.Wikipedia("en")
        page = wiki.page(query)
        if page.exists():
            return f"According to Wikipedia, {page.summary[:200]}..."
        return "No relevant Wikipedia page found for the query."
    except Exception as e:
        return f"Error checking Wikipedia: {str(e)}"


def initialize_agent_with_tools():
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        huggingfacehub_api_token=hf_api_token,
        temperature=0.7,
        max_length=512,
    )
    tools = [
        Tool(
            name="Wikipedia Fact Check",
            func=wikipedia_fact_check,
            description="Check facts using Wikipedia for a given query or claim.",
        )
    ]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    return agent


url = st.text_input(
    "Enter a news article URL:", placeholder="e.g., https://www.bbc.com/news/article"
)
if url:
    with st.spinner("Processing article..."):

        text = fetch_article(url)
        if text:
            article_hash = get_article_hash(text)
            if article_hash != st.session_state.article_hash:
                cached_vectorstore = load_cached_vectorstore(article_hash)
                if cached_vectorstore:
                    st.session_state.vectorstore = cached_vectorstore
                    st.session_state.article_hash = article_hash

                    llm = HuggingFaceEndpoint(
                        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                        huggingfacehub_api_token=hf_api_token,
                        temperature=0.7,
                        max_length=512,
                    )
                    summary_chain = load_summarize_chain(
                        llm=llm,
                        chain_type="map_reduce",
                        map_prompt=PromptTemplate.from_template(
                            "Summarize this in 50 words or less: {text}"
                        ),
                        combine_prompt=PromptTemplate.from_template(
                            "Combine these summaries into a concise overview (50-100 words): {text}"
                        ),
                    )
                    st.session_state.summary = summary_chain.run(
                        [Document(page_content=text)]
                    )
                else:
                    (
                        st.session_state.vectorstore,
                        st.session_state.article_hash,
                        st.session_state.summary,
                    ) = process_article(url)
            else:
                st.session_state.vectorstore = (
                    load_cached_vectorstore(article_hash)
                    or st.session_state.vectorstore
                )

            if st.session_state.vectorstore:
                st.session_state.qa_chain = initialize_qa_chain(
                    st.session_state.vectorstore
                )
                st.success("Article processed successfully!")
                st.markdown("**Summary:**")
                st.write(st.session_state.summary)

if st.session_state.qa_chain:
    with st.form("question_form"):
        question = st.text_input(
            "Ask a question or fact-check a claim:",
            placeholder="e.g., What is the main topic? Is this claim true?",
        )
        submitted = st.form_submit_button("Submit")

        if submitted and question:
            with st.spinner("Generating response..."):
                if (
                    "fact-check" in question.lower()
                    or "is this true" in question.lower()
                ):
                    agent = initialize_agent_with_tools()
                    response = agent.run(question)
                    st.markdown("**Fact-Check Result:**")
                    st.write(response)
                else:
                    result = st.session_state.qa_chain(
                        {
                            "question": question,
                            "chat_history": st.session_state.chat_history,
                        }
                    )
                    answer = result["answer"]
                    source_documents = result["source_documents"]

                    st.session_state.chat_history.append((question, answer))

                    st.markdown("**Answer:**")
                    st.write(answer)

                    with st.expander("Source Documents", expanded=False):
                        for i, doc in enumerate(source_documents, 1):
                            st.markdown(
                                f"**Chunk {i}** (Source: {doc.metadata.get('source', 'Unknown')}):"
                            )
                            st.write(f"{doc.page_content[:200]}...")


st.markdown("### Chat History")
if st.session_state.chat_history:
    for i, (q, a) in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"**Q{i}**: {q}")
        st.markdown(f"**A{i}**: {a}")
else:
    st.write("No questions asked yet.")
