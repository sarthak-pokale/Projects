import streamlit as st
import speech_recognition as sr
from transformers import pipeline
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from qdrant_client import QdrantClient
import tempfile
import PyPDF2

# Initialize components
model_name = "BAAI/bge-large-en"
embedding_model = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': False})

qdrant_url = "http://localhost:6333"
client = QdrantClient(url=qdrant_url, prefer_grpc=False)

# Function to load and process PDF
def process_pdf(pdf_file):
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    return texts

# Function to index documents into Qdrant
def index_documents(texts, collection_name):
    qdrant = Qdrant.from_documents(texts, embedding_model, url=qdrant_url, prefer_grpc=False, collection_name=collection_name)
    return qdrant

# Function to handle speech recognition
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio_data = recognizer.listen(source)
        st.info("Processing...")
        try:
            query = recognizer.recognize_google(audio_data)
            return query
        except sr.UnknownValueError:
            return "Could not understand the audio."
        except sr.RequestError:
            return "Could not request results from Google Speech Recognition service."

# Streamlit UI
st.title("Sarthak's RAG speech recognition model")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    texts = process_pdf(pdf_path)
    collection_name = "user_uploaded_docs"
    qdrant = index_documents(texts, collection_name)
    st.success("PDF file processed and indexed successfully!")

    if st.button("Record a question"):
        query = recognize_speech()
        st.write(f"Recognized question: {query}")

        if query and qdrant:
            docs = qdrant.similarity_search_with_score(query=query, k=3)
            if docs:
                st.write("Surely, here is your answer:")
                for doc, score in docs:
                    st.write({"score": score, "content": doc.page_content, "metadata": doc.metadata})
            else:
                st.write("Sorry, no relevant information found.")
