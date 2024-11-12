from typing import Any, Dict, List
from uuid import UUID
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, Language
from langchain.storage import LocalFileStore
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.vectorstores import Chroma
import streamlit as st

st.set_page_config(
    page_title="GeneralGPT",
    page_icon="😁",
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[]
if "history" not in st.session_state:
    st.session_state["history"] = []

def save_message(message, role, max_cache = 200):
    st.session_state["messages"].append({"message":message, "role":role})
    st.session_state["history"].append({"message":message, "role":role})
    if len(st.session_state["history"]) > max_cache:
        st.session_state["history"] = st.session_state["history"][(len(st.session_state["history"]) - max_cache):]
    # for message in st.session_state["messages"]:
    #     st.markdown(message["message"])

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

def embed_file():
    content = ''
    for message in st.session_state["history"]:
        content += message["role"] + ": " + message["message"]

    text_spliter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20, length_function=len,
                                                  add_start_index=True)
    splits = text_spliter.create_documents([content])
    print(len(splits))
    print(splits)

    # 使用字符切割
    # text_spliter_1 = CharacterTextSplitter(separator='。',
    #                                        chunk_size=200,
    #                                        chunk_overlap=20,
    #                                        length_function=len,
    #                                        add_start_index=True,
    #                                        is_separator_regex=False)
    # text = text_spliter_1.create_documents([content])

    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    return retriever

def send_message(message,role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message,role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ]
)

st.title("General GPT")
st.markdown("""
Welcome!

Use this chatbot to ask whatever questions you want to ask!

""")

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Answer the question using your own knowledge and the following context. The information in the context may be 
        useless, DO NOT rely on it! If you don't know the answer, answer you don't know, don't make anything up.

        Context: {context}     
        """,
    ),
    ("user", "{question}"),
])

send_message("Hello! What Can I do for you?", "ai", save=False)
paint_history()

message = st.chat_input("Please input what you want to ask...")

if message:
    send_message(message, "user")
    retriever = embed_file()

    chain = ({
                 "context": retriever | RunnableLambda(format_docs),
                 "question": RunnablePassthrough()
             } | prompt | llm)
    with st.chat_message("ai"):
        response = chain.invoke(message)