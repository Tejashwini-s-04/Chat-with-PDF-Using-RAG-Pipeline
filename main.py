import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')


# Initialize model and QA pipeline
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Define directories
PDF_DIR = "C:/task1/pdf_files"  # Update this path if needed
VECTOR_STORE_DIR = "vector_store"

# Ensure the vector store directory exists
if not os.path.exists(VECTOR_STORE_DIR):
    os.makedirs(VECTOR_STORE_DIR)

# Step 1: Extract text from PDFs
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text_data = []
    for page in reader.pages:
        text_data.append(page.extract_text())
    return text_data

# Step 2: Segment text into chunks
def segment_text_into_chunks(text, chunk_size=500):
    sentences = sent_tokenize(text)
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < chunk_size:
            chunk += sentence + " "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + " "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# Step 3: Generate embeddings
def generate_embeddings(chunks):
    return embedding_model.encode(chunks, convert_to_tensor=False)

# Step 4: Store embeddings in FAISS
def store_embeddings(embeddings, chunks):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(VECTOR_STORE_DIR, "index"))
    np.save(os.path.join(VECTOR_STORE_DIR, "chunks.npy"), np.array(chunks, dtype=object))
    print("Embeddings stored successfully.")

# Step 5: Query handling
def retrieve_relevant_chunks(query, index, chunks, top_k=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# Step 6: Generate response using LLM
def generate_response(context, query):
    return qa_pipeline({"context": " ".join(context), "question": query})

def main():
    all_chunks = []
    
    if not os.path.isdir(PDF_DIR):
        print(f"Error: {PDF_DIR} is not a directory.")
        return

    for pdf_file in os.listdir(PDF_DIR):
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        print(f"Processing file: {pdf_file}")
        pdf_text = extract_text_from_pdf(pdf_path)
        print(f"Extracted text from {pdf_file}:\n{pdf_text[:500]}")  # Print first 500 chars
        text_data = extract_text_from_pdf(pdf_path)
        for page_text in text_data:
            chunks = segment_text_into_chunks(page_text)
            all_chunks.extend(chunks)

    # Step 2: Embed and store chunks
    print("Generating embeddings...")
    embeddings = generate_embeddings(all_chunks)
    print("Storing embeddings in vector database...")
    store_embeddings(embeddings, all_chunks)

    # Load FAISS index and chunks
    index = faiss.read_index(os.path.join(VECTOR_STORE_DIR, "index"))
    chunks = np.load(os.path.join(VECTOR_STORE_DIR, "chunks.npy"), allow_pickle=True)

    # Query loop
    while True:
        user_query = input("\nEnter your query (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break

        print("Searching for relevant information...")
        relevant_chunks = retrieve_relevant_chunks(user_query, index, chunks)

        print("Generating response...")
        response = generate_response(relevant_chunks, user_query)
        print("\nResponse:", response['answer'])

if __name__ == "__main__":
    main()
