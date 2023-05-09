import streamlit as st


import os

st.title("🤖 Personalized Bot with fintech info 🧠 ")

st.markdown(
    """ 
        #### 🗨️ Chat with a bot with additional information 📜 
        """
)
#%%
os.environ['OPENAI_API_KEY'] = st.secrets.key
#%%
import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./db" # Optional, defaults to .chromadb/ in the current directory
))
#%%

#%%
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
global vectorstore
try:
    vectorstore = Chroma(collection_name="fomcpresconf", client=client, embedding_function=embeddings)
except:
    raise ValueError(f"vectorstore doesn't seem to load")

#%%
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI, LLMChain

from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import VectorStore


def get_chain(
    vectorstore: VectorStore
) -> ConversationalRetrievalChain:
    """Create a ConversationalRetrievalChain for question/answering."""
    # Construct a ConversationalRetrievalChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation

    question_gen_llm = OpenAI(
        temperature=0,
        verbose=True,
    )
    streaming_llm = OpenAI(
        streaming=True,
        verbose=True,
        temperature=0,
    )

    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT
    )
    doc_chain = load_qa_chain(
        streaming_llm, chain_type="stuff", prompt=QA_PROMPT
    )

    qa = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        verbose=True
    )
    return qa

#%%
chat_history = []
qa_chain = get_chain(vectorstore)
#%%
wtf = st.text_input(
        "**What's on your mind?**",
        placeholder="Ask me anything from {}"
    )

if wtf:
        with st.spinner(
                "Generating Answer to your Query : `{}` ".format(wtf)
        ):
            res = qa_chain(
                {"question": wtf, "chat_history": chat_history}
            )

            st.info(res["answer"], icon="🤖")
