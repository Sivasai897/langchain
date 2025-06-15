# Import necessary libraries
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import FireCrawlLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables from .env file
# This is crucial for accessing the FireCrawl API key securely
print("Loading environment variables...")
load_dotenv()

# Get the FireCrawl API key from environment variables
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
if not FIRECRAWL_API_KEY:
    raise ValueError("FIRECRAWL_API_KEY not found in environment variables")

# Set up directory paths for storing the vector database
print("Setting up directory structure...")
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_firecrawl_db")

def create_firecrawl_chrome_db():
    """
    Creates a vector database from web content using FireCrawl.
    FireCrawl is a powerful web scraping tool that can handle:
    - JavaScript-rendered content
    - Dynamic web pages
    - Complex website structures
    """
    print("\nInitializing FireCrawl loader...")
    # Initialize FireCrawl loader with API key and target URL
    # mode="scrape" indicates we want to extract content from the page
    loader = FireCrawlLoader(
        api_key=FIRECRAWL_API_KEY,
        url="https://www.apple.com",
        mode="scrape"
    )
    
    print("Loading documents from FireCrawl...")
    docs = loader.load()
    print(f"Successfully loaded {len(docs)} documents")

    # Process metadata to handle list values
    print("\nProcessing document metadata...")
    for doc in docs:
        for key, value in doc.metadata.items():
            if isinstance(value, list):
                # Convert list values to strings for better storage
                doc.metadata[key] = "\n".join(value)

    # Split documents into smaller chunks
    print("\nSplitting documents into chunks...")
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,  # Size of each text chunk
        chunk_overlap=200  # Overlap between chunks to maintain context
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"Created {len(split_docs)} chunks from {len(docs)} documents")

    # Initialize embedding model
    print("\nInitializing embedding model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create and persist the vector database
    print("\nCreating vector database...")
    db = Chroma.from_documents(split_docs, embeddings, persistent_directory)
    print("Vector database created and persisted successfully")

# Check if database exists, if not create it
print("\nChecking for existing database...")
if not os.path.exists(persistent_directory):
    print("Database not found. Creating new database...")
    create_firecrawl_chrome_db()
else:
    print("Existing database found.")

# Initialize embedding model for querying
print("\nInitializing embedding model for querying...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load the existing database
print("Loading vector database...")
db = Chroma(persistent_directory, embeddings)

# Create a retriever for semantic search
print("\nSetting up retriever...")
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,  # Number of documents to retrieve
        "score_threshold": 0.5  # Minimum similarity score threshold
    }
)

# Perform the search query
print("\nPerforming semantic search...")
result = retriever.invoke("What is the latest news on Apple's products?")
print(f"Found {len(result)} relevant documents")

# Display the results
print("\nDisplaying search results:")
for i, doc in enumerate(result, start=1):
    print(f"\nDocument {i}:")
    print(doc.page_content)
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print("\n")
    print("-"*100)
      
       
    
    
        
