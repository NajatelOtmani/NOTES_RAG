import os
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DATA_DIR = "./data"
DB_DIR = "./chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_and_chunk_docs(data_dir):
  all_chunks = []
  for file in os.listdir(data_dir):
    if file.endswith(".pdf"):
      path = os.path.join(data_dir, file)
      reader = PdfReader(path)
      text = ""
      for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
          text += extracted + "\n"

      # Chunking
      start = 0
      while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        # Ripping out tiny blank chunks
        if len(chunk.strip()) > 50:
          all_chunks.append({"source": file, "text": chunk.strip()})
        start += CHUNK_SIZE - CHUNK_OVERLAP
  return all_chunks


def setup_vector_db():
  print("Initializing ChromaDB and embedding model...")
  model = SentenceTransformer("all-MiniLM-L6-v2")

  client = chromadb.PersistentClient(path=DB_DIR)
  collection_name = "notes_rag"

  try:
    client.delete_collection(collection_name)
  except:
    pass

  collection = client.create_collection(name=collection_name)

  chunks = load_and_chunk_docs(DATA_DIR)
  print(f"Embedding {len(chunks)} chunks into ChromaDB...")

  texts = [c["text"] for c in chunks]
  metadatas = [{"source": c["source"]} for c in chunks]
  ids = [str(i) for i in range(len(chunks))]

  embeddings = model.encode(texts).tolist()

  collection.add(
      embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids
  )
  print("Database built successfully!")
  return collection


def test_retrieval(collection, query, top_k=3):
  model = SentenceTransformer("all-MiniLM-L6-v2")
  query_embedding = model.encode([query]).tolist()

  results = collection.query(
      query_embeddings=query_embedding, n_results=top_k
  )

  print(f"\nQuery: '{query}'")
  print("-" * 60)
  for idx, (doc, meta) in enumerate(
      zip(results["documents"][0], results["metadatas"][0]), 1
  ):
    print(f"Result {idx} (Source: {meta['source']})")
    print(doc[:300] + "...")
    print("-" * 40)


if __name__ == "__main__":
  collection = setup_vector_db()
  sample_question = "How do I handle missing data and outliers in machine learning?"
  test_retrieval(collection, sample_question, top_k=3)