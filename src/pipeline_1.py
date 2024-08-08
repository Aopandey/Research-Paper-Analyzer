import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Retrieve the Google API key from environment variables
os.getenv("GOOGLE_API_KEY")

# Configure the Generative AI API with the retrieved API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Function to extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        # Loop through each page in the PDF and extract text
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Function to split the extracted text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# Function to generate vector embeddings from text chunks and store them in a FAISS index
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # Save the FAISS index locally for later retrieval
    vector_store.save_local("faiss_index")


# Function to create a conversational chain for answering questions based on context
def get_conversational_chain():
    # Define a prompt template for the conversational model
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    # Configure the conversational AI model with specific parameters
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)

    # Initialize the prompt template with input variables
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Load a question-answering chain using the configured model and prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


# Function to handle user input and generate a response based on the FAISS index
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load the FAISS index from local storage
    new_db = FAISS.load_local("../faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Perform a similarity search in the FAISS index based on the user's question
    docs = new_db.similarity_search(user_question)

    # Get the conversational chain for generating a response
    chain = get_conversational_chain()

    # Generate a response based on the retrieved documents and user question
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    # Output the response in the Streamlit app
    print(response)
    st.write("Reply: ", response["output_text"])


# Main function to define the Streamlit app's interface
def main():
    st.set_page_config("Chat PDF")  # Set the title of the Streamlit app
    st.header("Chat with PDF using Gemini (Langchain)")  # Display the app header

    # Text input for the user to ask a question
    user_question = st.text_input("Ask a Question from the PDF Files")

    # If a question is provided, process the input and generate a response
    if user_question:
        user_input(user_question)

    # Sidebar for file upload and processing options
    with st.sidebar:
        st.title("Menu:")
        # File uploader for multiple PDF files
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True)
        # Button to trigger processing of the uploaded PDF files
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Extract text from the uploaded PDFs, split it into chunks, and store it in the FAISS index
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")  # Indicate that processing is complete


# Entry point to start the Streamlit app
if __name__ == "__main__":
    main()
