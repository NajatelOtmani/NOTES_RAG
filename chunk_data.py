import os
from pypdf import PdfReader

DATA_DIR = "./data"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_and_clean_pdfs(data_dir):
  documents = []
  for file in os.listdir(data_dir):
    if file.endswith(".pdf"):
      path = os.path.join(data_dir, file)
      reader = PdfReader(path)
      text = ""
      for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
          text += extracted + "\n"

      documents.append({"filename": file, "content": text})
  return documents


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
  chunks = []
  start = 0
  while start < len(text):
    end = start + chunk_size
    chunk = text[start:end]
    chunks.append(chunk)
    start += chunk_size - overlap
  return chunks


# --- Run & Inspect ---
if __name__ == "__main__":
  docs = load_and_clean_pdfs(DATA_DIR)
  all_chunks = []

  for doc in docs:
    raw_chunks = chunk_text(doc["content"])
    for i, chunk in enumerate(raw_chunks):
      all_chunks.append({
          "source": doc["filename"],
          "chunk_id": f"{doc['filename']}_chunk_{i}",
          "text": chunk.strip(),
      })

  print(f"Total PDFs processed: {len(docs)}")
  print(f"Total chunks created: {len(all_chunks)}\n")

  print("=== SAMPLE CHUNKS ===")
  import random

  sample = random.sample(all_chunks, min(3, len(all_chunks)))
  for idx, s in enumerate(sample, 1):
    print(f"--- Chunk {idx} (Source: {s['source']}) ---")
    print(s["text"])
    print("-" * 50)