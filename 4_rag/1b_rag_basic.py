import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Set up paths and initialize the embedding model
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Initialize the embedding model for converting text to vectors
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load the existing vector store from the persistent directory
vectorstore = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Define the query to search for in the vector store
query = "Who is Odysseus' wife?"

# Configure the retriever with similarity search parameters
# search_type: Uses similarity score threshold to filter results
# k: Number of most similar documents to retrieve
# score_threshold: Minimum similarity score (0.5 = 50% similarity)
retriever = vectorstore.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {"k": 3, "score_threshold": 0.5}
)

# Retrieve relevant documents based on the query
relevant_docs = retriever.get_relevant_documents(query)

# Display the retrieved documents with their content and metadata
for i,doc in enumerate(relevant_docs, start=1):
    print(f"Relevant Document {i}:")
    print(doc.page_content)
    print("\n") 
    if doc.metadata.get("source"):
        print(f"Source: {doc.metadata['source']}")
    print("-"*100)


