import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.vectorstores import FAISS

# Load environment variables from 'api.env'
load_dotenv()
GOOGLE_API_KEY = os.getenv("Google_API_key")

def get_pdf_text(pdf_file_name):
    """Reads text from a PDF file."""
    text = ""
    pdf_reader = PdfReader(pdf_file_name)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Generates embeddings and creates a FAISS vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Defines a conversational question-answering chain."""
    prompt_template = """
    Answer the question as detailed points as possible from the provided context, make sure to provide all the details, if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    """Handles user input, performs similarity search, and generates a response."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("Reply: ", response["output_text"])

def main():
    """Main function to set up Streamlit app and process user interactions."""
    st.set_page_config(page_title="HistoryChat App", page_icon="📜", layout="wide")

    st.title("Indian History Chat with Your Virtual Teacher")
    st.markdown("---")

    # Assuming "history.pdf" is the fixed file name
    pdf_file_name = "history.pdf"
    
    with st.spinner("Processing..."):
        raw_text = get_pdf_text(pdf_file_name)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
    
    st.success("Done")

    user_question = st.text_input("Ask a Question related to Indian History (and Please wait for 15-20 secs for result)")

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
