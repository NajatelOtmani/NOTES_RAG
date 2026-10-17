"""
make_eval_set.py — generates eval_set.json from YOUR indexed PDFs.

    python make_eval_set.py

For each PDF it samples a few passages from ChromaDB and asks llama3.2 to write a question
that the passage answers. The PDF name is stored as the expected source, so there are no
typos in file names. Ollama must be running. Review the result and delete weak questions.
"""

import json
import random
import re
from collections import defaultdict

import chromadb
import ollama

DB_DIR = "./chroma_db"
COLLECTION_PREFERENCE = ["notes_rag_500", "notes_rag"]
OLLAMA_MODEL = "llama3.2"
QUESTIONS_PER_PDF = 3
MIN_CHARS = 250            # skip tiny passages
EXCLUDE = []               # e.g. ["CV"] to skip PDFs whose name contains "CV"
OUT_FILE = "eval_set.json"

PROMPT = """Below is a passage from a study document.
Write ONE clear question that a student could ask and that this passage answers.
Rules: do not copy sentences from the passage, do not mention "the passage" or "the text",
and output only the question, nothing else.

Passage:
{passage}

Question:"""

random.seed(42)
client = chromadb.PersistentClient(path=DB_DIR)
names = [c if isinstance(c, str) else c.name for c in client.list_collections()]
coll_name = next((n for n in COLLECTION_PREFERENCE if n in names), None)
if coll_name is None:
    raise SystemExit("No collection found. Run build_index.py first.")

data = client.get_collection(coll_name).get(include=["documents", "metadatas"])
by_source = defaultdict(list)
for doc, meta in zip(data["documents"], data["metadatas"]):
    if len(doc) >= MIN_CHARS:
        by_source[meta["source"]].append(doc)

cases = []
for source, passages in sorted(by_source.items()):
    if any(x.lower() in source.lower() for x in EXCLUDE):
        print(f"skip   {source}")
        continue
    picked = random.sample(passages, min(QUESTIONS_PER_PDF, len(passages)))
    for passage in picked:
        out = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": PROMPT.format(passage=passage)}],
            options={"temperature": 0.5, "num_predict": 60},
        )["message"]["content"]
        question = re.sub(r"\s+", " ", out).strip().strip('"')
        if question.endswith("?") and 15 < len(question) < 200:
            cases.append({"question": question, "expected_source": source})
            print(f"ok     {source}: {question}")
        else:
            print(f"reject {source}: {question[:60]}")

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(cases, f, ensure_ascii=False, indent=2)
print(f"\n{len(cases)} questions written to {OUT_FILE} (from collection {coll_name})")