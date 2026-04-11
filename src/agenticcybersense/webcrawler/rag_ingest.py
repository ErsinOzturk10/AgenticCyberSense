from typing import List
import math
from sentence_transformers import SentenceTransformer
import chromadb

EMBED_MODEL = SentenceTransformer("all-mpnet-base-v2")
CHUNK_SIZE = 1000  # characters

client = chromadb.Client()
collection = client.get_or_create_collection(name="webcrawler_pages")

def chunk_text(text: str, size: int = CHUNK_SIZE) -> List[str]:
    chunks = []
    for i in range(0, len(text), size):
        chunks.append(text[i : i + size])
    return chunks

def embed_texts(texts: List[str]) -> List[List[float]]:
    return [EMBED_MODEL.encode(t).tolist() for t in texts]

def upsert_site(url: str, page_idx: int, title: str, content: str, metadata: dict):
    chunks = chunk_text(content)
    embeddings = embed_texts(chunks)
    docs = [{"url": url, "page_idx": page_idx, "title": title, **metadata} for _ in chunks]
    ids = [f"{url}::{page_idx}::{i}" for i, _ in enumerate(chunks)]
    collection.upsert(
        documents=chunks,
        metadatas=docs,
        ids=ids,
        embeddings=embeddings,
    )