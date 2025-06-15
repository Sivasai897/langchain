import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Set up paths and initialize the embedding model
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# Initialize the embedding model for converting text to vectors
embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

# Load the existing vector store with metadata
# This vector store contains documents with source information in their metadata
vectorstore = Chroma(persist_directory = persistent_directory, embedding_function = embeddings)

# Define the query to search for in the vector store
query = "What is the name of the character who is the son of Zeus and Leda?"
print(f"\nQuery: {query}")

# Configure the retriever with similarity search parameters
# search_type: Uses similarity score threshold to filter results
# k: Number of most similar documents to retrieve (3 in this case)
# score_threshold: Minimum similarity score (0.5 = 50% similarity)
# This helps ensure we only get relevant results
retriever = vectorstore.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {"k": 3, "score_threshold": 0.5}
)

# Retrieve relevant documents based on the query
# The results will include both the content and metadata (source information)
print("\nRetrieving relevant documents...")
relevant_docs = retriever.get_relevant_documents(query)
print(f"Found {len(relevant_docs)} relevant documents\n")

# Display the retrieved documents with their content and source metadata
# This helps verify the source of each piece of information
for i,doc in enumerate(relevant_docs, 1):
    print(f"Relevant Document {i}:")
    print(doc.page_content)
    print("\n")
    if doc.metadata.get("source"):
        print(f"Source: {doc.metadata['source']}")
    print("-"*100)

