"""
build_index.py — builds one ChromaDB collection per chunk size for CORTEX-V.

    python build_index.py

Creates: notes_rag_200, notes_rag_500, notes_rag_1000 in ./chroma_db
(the CORTEX-V chunk-size switch lists whichever of these exist).
"""

import os

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DATA_DIR = "./data"
DB_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZES = [200, 500, 1000]
OVERLAP_RATIO = 0.10  # overlap = 10% of the chunk size (20 / 50 / 100 chars)
MIN_CHARS = 50
BATCH = 500


def read_pdfs(data_dir):
    """Extract each PDF's text once; it's reused for every chunk size."""
    docs = {}
    for file in sorted(os.listdir(data_dir)):
        if file.lower().endswith(".pdf"):
            reader = PdfReader(os.path.join(data_dir, file))
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            docs[file] = text
    return docs


def chunk_text(docs, size):
    overlap = int(size * OVERLAP_RATIO)
    step = size - overlap
    chunks = []
    for source, text in docs.items():
        for start in range(0, len(text), step):
            piece = text[start : start + size].strip()
            if len(piece) > MIN_CHARS:
                chunks.append({"source": source, "text": piece})
    return chunks


def build(client, model, docs, size):
    name = f"notes_rag_{size}"
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(name=name)

    chunks = chunk_text(docs, size)
    texts = [c["text"] for c in chunks]
    print(f"[{name}] embedding {len(chunks)} chunks…")
    # normalized vectors → L2 ranking == cosine ranking
    embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True).tolist()

    for i in range(0, len(chunks), BATCH):
        j = i + BATCH
        collection.add(
            ids=[f"{size}-{k}" for k in range(i, min(j, len(chunks)))],
            embeddings=embeddings[i:j],
            documents=texts[i:j],
            metadatas=[{"source": c["source"]} for c in chunks[i:j]],
        )
    print(f"[{name}] done.\n")


if __name__ == "__main__":
    docs = read_pdfs(DATA_DIR)
    print(f"Found {len(docs)} PDFs in {DATA_DIR}")
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=DB_DIR)
    for size in CHUNK_SIZES:
        build(client, model, docs, size)
    print("All collections built.")