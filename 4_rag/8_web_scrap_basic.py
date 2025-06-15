# Import necessary libraries
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os

# Load environment variables from .env file
# This is important for securely managing API keys and other sensitive configurations
load_dotenv()

# Set up directory paths for storing the vector database
# This ensures the database is stored in a consistent location relative to the script
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir,"db")
persistent_directory = os.path.join(db_dir, "chroma_web_scrap_db")

# Initialize the embedding model
# GoogleGenerativeAIEmbeddings is used to convert text into vector representations
# These vectors capture semantic meaning and enable similarity searches
embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

# Initialize the web scraper
# WebBaseLoader is a LangChain utility that handles web page loading and parsing
# It automatically extracts text content from HTML while removing unnecessary elements
loader = WebBaseLoader("https://www.apple.com")
documents = loader.load()

# Split the documents into smaller chunks
# This is crucial for:
# 1. Managing token limits of embedding models
# 2. Creating more focused and relevant search results
# 3. Enabling more precise retrieval of specific information
text_splitter = CharacterTextSplitter(
    chunk_size = 1000,  # Size of each text chunk
    chunk_overlap = 200  # Overlap between chunks to maintain context
)
docs = text_splitter.split_documents(documents)

# Initialize or load the vector database
# Chroma is a vector store that:
# 1. Stores document embeddings for efficient similarity search
# 2. Persists data to disk for reuse across sessions
# 3. Enables semantic search capabilities
if not os.path.exists(persistent_directory):
    # Create new database if it doesn't exist
    db = Chroma.from_documents(docs, embeddings, persistent_directory)
else:
    # Load existing database
    db = Chroma(persistent_directory, embeddings)

# Create a retriever for searching the vector database
# This configuration:
# 1. Uses similarity score threshold to ensure quality of results
# 2. Returns top 3 most relevant documents
# 3. Only returns documents with similarity score above 0.5
retriever = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {
        "k": 3,  # Number of documents to retrieve
        "score_threshold": 0.5  # Minimum similarity score threshold
    }
)

# Perform a semantic search query
# The retriever will find documents most relevant to the query
# based on semantic similarity rather than just keyword matching
relevant_docs = retriever.invoke("What is the latest news on Apple's products?")

# Display the retrieved documents with their metadata
# This helps in:
# 1. Understanding the relevance of each document
# 2. Tracking the source of information
# 3. Evaluating the quality of the retrieval
for i, doc in enumerate(relevant_docs, start=1):
    print(f"\nDocument {i}:")
    print(doc.page_content)
    if doc.metadata:
        print(f"\nSource: {doc.metadata.get('source', 'Unknown')}")
    print("\n")
    print("-"*100)


