# import databutton as db
import streamlit as st
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from llama_index import StorageContext, load_index_from_storage
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
import os
# user = db.user.get()
# name = user.name if user.name else "you"


# user = db.user.get()
# name = user.name if user.name else "you"

st.title("🤖 Personalized Bot with Memory 🧠 ")

st.markdown(
    """ 
        #### 🗨️ Chat with a bot with additional information 📜 with `Conversational Buffer Memory`  
        > *powered by [LangChain]('https://langchain.readthedocs.io/en/latest/modules/memory.html#memory') + 
        [OpenAI]('https://platform.openai.com/docs/models/gpt-3-5') + [Streamlit](https://docs.streamlit.io/) + [LlamaIndex](https://gpt-index.readthedocs.io/en/stable/index.html)*
        ----
        """
)

option = st.selectbox(
    'Which data do you want to use?',
    ('Finite-size effects of avalanche dynamics', 'A Review of ChatGPT Applications'))

st.write('You selected:', option)
os.environ['OPENAI_API_KEY'] = st.secrets.key
if option:
    if option == 'Finite-size effects of avalanche dynamics':
        storage_context = StorageContext.from_defaults(persist_dir="./storage1")

    if option == 'A Review of ChatGPT Applications':
        storage_context = StorageContext.from_defaults(persist_dir="./storage")

    index = load_index_from_storage(storage_context)
    tools = [
        Tool(
            name="GPT Index",
            func=lambda q: str(index.as_query_engine().query(q)),
            description="useful for when you want to answer questions about the author. The input to this tool should be a complete english sentence.",
            return_direct=True
        ),
    ]
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history"
        )
    llm = ChatOpenAI(temperature=0)
    agent_chain = initialize_agent(tools, llm, agent="conversational-react-description", memory=st.session_state.memory)
    wtf = st.text_input(
        "**What's on your mind?**",
        placeholder="Ask me anything from {}"
    )

    if wtf:
        with st.spinner(
                "Generating Answer to your Query : `{}` ".format(wtf)
        ):
            res = agent_chain.run(input=wtf)
            st.info(res, icon="🤖")
    with st.expander("History/Memory"):
        st.session_state.memory
    if st.button('forget the context.'):
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history"
        )
