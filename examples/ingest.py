"""
Ingestion pipeline: create Milvus collection and populate it with embeddings.

Run from repo root: python examples/ingest.py
"""

from langchain_community.document_loaders import DirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from tqdm import tqdm

from src.bge_embeddigns import bge_embeddings, embed_bge
from src.splade_embeddings import embed_splade
from src.utils import reformat_text
from src.vectorstore import activate_collection, create_collection

DOCS_DIR = "docs/youtube_test"

# ---------------------------------------------------------------------------
# 1. Create collection (no-op if it already exists)
# ---------------------------------------------------------------------------
create_collection()
collection = activate_collection()

# ---------------------------------------------------------------------------
# 2. Load and clean transcripts
# ---------------------------------------------------------------------------
loader = DirectoryLoader(DOCS_DIR, glob="*.txt", show_progress=True)
docs = loader.load()
print(f"Loaded {len(docs)} documents from {DOCS_DIR}")

for doc in docs:
    doc.page_content = reformat_text(doc.page_content)

# ---------------------------------------------------------------------------
# 3. Split into chunks
# ---------------------------------------------------------------------------
semantic_chunker = SemanticChunker(bge_embeddings, breakpoint_threshold_type="standard_deviation")
chunks = semantic_chunker.create_documents([doc.page_content for doc in docs])
print(f"Split into {len(chunks)} chunks")

# ---------------------------------------------------------------------------
# 4. Embed and insert
# ---------------------------------------------------------------------------
collection.load()
for chunk in tqdm(chunks, desc="Embedding and inserting"):
    text = chunk.page_content
    dense_emb = embed_bge(text)
    sparse_emb = embed_splade(text)
    sparse_dict = {
        int(i): float(v) for i, v in zip(sparse_emb.indices, sparse_emb.data)
    }
    entity = [[text], [sparse_dict], [dense_emb]]
    collection.insert(entity)

collection.flush()
print("Ingestion complete.")
