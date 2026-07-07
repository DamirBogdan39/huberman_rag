"""
HyDE RAG pipeline debug script.

Migrated from notebooks/hyde_rag.ipynb.
Run from repo root: python examples/hyde_rag.py
"""
import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_classic.chains import HypotheticalDocumentEmbedder
from langchain_core.prompts import PromptTemplate

from config.models import models
from src.bge_embeddigns import bge_embeddings
from src.vectorstore import hyde_query

load_dotenv()

# ---------------------------------------------------------------------------
# 1. LLM
# ---------------------------------------------------------------------------
llm = ChatAnthropic(model=models.sonnet_4_6)
print(f"[LLM] model: {models.sonnet_4_6}")

# ---------------------------------------------------------------------------
# 2. HyDE prompt & embedder
# ---------------------------------------------------------------------------
HYDE_PROMPT = (
    "Answer the question in the style of doctor Andrew Huberman.\n"
    "Question: {question}\nAnswer:"
)

hyde_prompt = PromptTemplate.from_template(template=HYDE_PROMPT)
print(f"[Prompt] template:\n{hyde_prompt.template}\n")

embeddings = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=bge_embeddings,
    custom_prompt=hyde_prompt,
)

# Inspect the underlying chain (llm_chain is a RunnableSequence in modern langchain)
print(f"[HyDE] llm_chain type: {type(embeddings.llm_chain)}")
print(f"[HyDE] base_embeddings: {type(embeddings.base_embeddings).__name__}")

# ---------------------------------------------------------------------------
# 3. embed_query — generates a hypothetical doc then embeds it
# ---------------------------------------------------------------------------
TEST_QUERY = "Which compounds are in coffee?"
print(f"\n[embed_query] query: {TEST_QUERY!r}")

vector = embeddings.embed_query(TEST_QUERY)
print(f"[embed_query] vector dim: {len(vector)}")
print(f"[embed_query] first 5 values: {vector[:5]}")

# ---------------------------------------------------------------------------
# 4. Full HyDE retrieval from Milvus
# ---------------------------------------------------------------------------
RETRIEVAL_QUERY = "How to do clinical trials with placebo?"
print(f"\n[hyde_query] query: {RETRIEVAL_QUERY!r}")

results = hyde_query(RETRIEVAL_QUERY)
print(f"[hyde_query] hits returned: {len(results[0])}")
for i, hit in enumerate(results[0]):
    content = hit.get("entity", {}).get("page_content", "")
    print(f"  [{i}] score={hit.get('distance', 'N/A'):.4f}  text={content[:120]!r}")
