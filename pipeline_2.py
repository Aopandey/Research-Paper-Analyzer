import os
import requests
from PyPDF2 import PdfReader
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from a .env file
load_dotenv()

# Configure Google Generative AI with the API key from environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Function to download a PDF file from a given URL and save it locally
def download_pdf(url, save_path):
    response = requests.get(url)
    response.raise_for_status()  # Check if the download was successful
    with open(save_path, "wb") as file:
        file.write(response.content)  # Save the content of the PDF to a file


# Function to extract text from a PDF file
def load_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()  # Extract text from each page and concatenate
    return text


# Function to split extracted text into chunks of a specified maximum length
def split_text_recursively(text, max_length=1000, chunk_overlap=0):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + max_length

        # Ensure chunks end at a word boundary
        if end < text_length:
            end = text.rfind(' ', start, end) + 1
            if end <= start:
                end = start + max_length

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)  # Add the chunk to the list of chunks

        start = end - chunk_overlap  # Move to the next chunk, considering overlap
        if start >= text_length:
            break

    return chunks


# Setup ChromaDB client with Google Generative AI embedding function
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=os.getenv("GOOGLE_API_KEY"))
client = chromadb.PersistentClient(path="embeddings/gemini")

# Create or retrieve a collection in ChromaDB for storing PDF chunks
collection = client.get_or_create_collection(name="pdf_rag", embedding_function=google_ef)


# Function to add text chunks to the ChromaDB collection
def add_chunks_to_collection(chunks):
    for i, d in enumerate(chunks):
        collection.add(documents=[d], ids=[str(i)])  # Add each chunk with a unique ID


# Function to build a single string from the list of relevant context chunks
def build_escaped_context(context):
    escaped_context = ""
    for item in context:
        escaped_context += item + "\n\n"  # Combine chunks with spacing for readability
    return escaped_context


# Function to find the most relevant context for a given query using ChromaDB
def find_relevant_context(query, db, n_results=3):
    results = db.query(query_texts=[query], n_results=n_results)  # Query ChromaDB
    escaped_context = build_escaped_context(results['documents'][0])  # Build a context string
    return escaped_context


# Function to create a prompt for Gemini to generate an answer
def create_prompt_for_gemini(query, context):
    prompt = f"""
    You are a helpful agent that answers questions using the text from the context below.
    Both the question and the context is shared with you and you should answer the
    question basis the context. If the context does not have enough information
    for you to answer the question correctly, inform about the absence of relevant
    context as part of your answer.

    Question : {query}
    \n
    Context : {context}
    \n
    Answer :
    """
    return prompt


# Function to generate an answer using the Gemini model from a given prompt
def generate_answer_from_gemini(prompt):
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    result = model.generate_content(prompt)  # Generate a response based on the prompt
    return result


# Streamlit UI for interacting with the system
def main():
    st.set_page_config("Chat PDF")  # Set the title of the Streamlit app
    st.header("Chat with PDF using Gemini (ChromaDB)")  # Display the app header

    # Input box for users to ask questions about the PDF content
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        # Find relevant context, create a prompt, and generate an answer
        context = find_relevant_context(user_question, collection)
        prompt = create_prompt_for_gemini(user_question, context)
        answer = generate_answer_from_gemini(prompt)
        st.write("Reply: ", answer.text)  # Display the answer in the app

    with st.sidebar:
        st.title("Menu:")

        # File uploader for multiple PDF files
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True)

        # Button to start the processing of uploaded PDF files
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Ensure directory exists for saving uploaded PDFs
                if not os.path.exists("uploaded_pdfs"):
                    os.makedirs("uploaded_pdfs")

                # Process each uploaded PDF file
                for pdf in pdf_docs:
                    pdf_path = os.path.join("uploaded_pdfs", pdf.name)
                    with open(pdf_path, "wb") as f:
                        f.write(pdf.getbuffer())  # Save the uploaded PDF

                    raw_text = load_pdf(pdf_path)  # Extract text from the PDF
                    text_chunks = split_text_recursively(raw_text, max_length=2000, chunk_overlap=200)
                    add_chunks_to_collection(text_chunks)  # Add the text chunks to the collection

                st.success("Done")  # Indicate that processing is complete


# Entry point to start the Streamlit app
if __name__ == "__main__":
    main()
