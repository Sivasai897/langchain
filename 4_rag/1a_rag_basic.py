import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

# Set up file paths for the text file and vector database
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Initialize vector database if it doesn't exist
if not os.path.exists(persistent_directory):
    print("Creating database in", persistent_directory)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load and process the text file
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()

    # Split text into smaller chunks for better processing
    # chunk_size: number of characters per chunk
    # chunk_overlap: number of characters to overlap between chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    print(f"Found {len(documents)} documents")
    print("Splitting documents into chunks...")

    # Create embeddings and store them in Chroma vector database
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
else:
    print("Vector store already exists. No need to initialize.")
        
