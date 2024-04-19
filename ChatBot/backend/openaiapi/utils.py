from openai import OpenAI
from django.conf import settings
import os

from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage

os.environ['OPENAI_API_KEY'] = settings.API_KEY

#define llm model that will be used
llm = ChatOpenAI(model="gpt-3.5-turbo")

#load the data
loader=TextLoader('YOUR_FILE_HERE.txt')
docs = loader.load()

#split the document_loaders
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

#embeddig
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve
retriever = vectorstore.as_retriever()

# create history aware retriever.
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

#history chain
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, please think rationally and answer from your own knowledge base. \
Use the following pieces of context to answer the question at the end.


{context}
"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)



chat_history = []



def chat_gpt(message):
    question = message
    print(question)
    ai_msg= rag_chain.invoke({"input": question, "chat_history": chat_history})

    chat_history.extend([HumanMessage(content=question), ai_msg["answer"]])
    return ai_msg["answer"]
