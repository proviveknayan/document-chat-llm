import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter ## to split the text into smaller chunks
from langchain.embeddings import HuggingFaceEmbeddings ## to create embeddings from the chunks
from langchain.vectorstores import FAISS ## to create vector store for the embeddings
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory ## to remember the running context of a chat
from langchain.chains import ConversationalRetrievalChain ## to get chat histry and reply to queries

## get the list of PDF(s) in the root directory
pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]

## function to get the text from the PDF(s)
def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

## function to split the text into smaller chunks
def get_text_chunks(pdf_text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    text_chunks = text_splitter.split_text(pdf_text)
    return text_chunks

## function to create vector store for the chunks
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

## function to create conversation chain
def get_chat_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(chat_memory='chat_history', return_messages=True)
    chat_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return chat_chain

def main():
    st.set_page_config(page_title="Chat With Your PDF(s)", page_icon=":sunglasses:")

    st.header("Chat With Your PDF(s) :sunglasses:")
    st.text_input("Ask a question to your document(s):")
    
    with st.sidebar:
        st.subheader("Your Documents")
        if len(pdf_files) == 0:
            st.write('Found 0 PDF file. Please add one to the root directory')
        else:
            st.write(f'Found {len(pdf_files)} PDF file(s).')
            st.write('- ' + '\n- '.join(pdf_files))
            with st.spinner("Processing the PDF(s)"):

                ## get the text from PDF(s)
                pdf_text = get_pdf_text(pdf_files)
                #st.write(pdf_text)

                ## split the text into smaller chunks
                text_chunks = get_text_chunks(pdf_text)
                #st.write(text_chunks)

                ## create vector store for the chunks
                vector_store = get_vector_store(text_chunks)

                ## create conversation chain
                load_dotenv()
                st.session_state.chat_chain = get_chat_chain(vector_store)

if __name__ == '__main__':
    main()