import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Set up directory paths for books and vector database
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# Initialize vector database if it doesn't exist
if not os.path.exists(persistent_directory):
    print("Creating database in", persistent_directory)

    if not os.path.exists(books_dir):
        raise FileNotFoundError(f"Books directory not found: {books_dir}")
    
    # Get all text files from the books directory
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]
    print(f"\nFound {len(book_files)} text files in the books directory:")
    for file in book_files:
        print(f"- {file}")

    # Process each book file and add metadata
    # Metadata helps track the source of each text chunk and enables filtering/searching by source
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        print(f"\nProcessing: {book_file}")
        loader = TextLoader(file_path, encoding="utf-8")
        book_docs = loader.load()
        # Add source metadata to each document
        # This metadata will be useful for:
        # 1. Tracking which book each chunk came from
        # 2. Filtering results by source
        # 3. Providing context in search results
        for doc in book_docs:
            doc.metadata = {"source": book_file}
            documents.append(doc)
        print(f"Added {len(book_docs)} chunks from {book_file}")
    
    # Split documents into smaller chunks for better processing
    # chunk_size: number of characters per chunk
    # chunk_overlap: number of characters to overlap between chunks
    # Overlap helps maintain context between chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Display processing information
    print(f"\nProcessing Summary:")
    print(f"- Total documents before splitting: {len(documents)}")
    print(f"- Total chunks after splitting: {len(docs)}")
    print(f"- Average chunks per document: {len(docs)/len(documents):.2f}")

    # Create embeddings and store them in Chroma vector database
    # The embeddings will preserve the metadata for each chunk
    print("\nCreating embeddings and storing in vector database...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print("Vector database created successfully!")
else:
    print("Vector store already exists. No need to initialize.")
    
