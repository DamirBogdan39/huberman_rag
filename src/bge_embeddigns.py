from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from src.device import get_device

model_name = "BAAI/bge-large-en-v1.5"

# True - computes cosine similarity
encode_kwargs = {"normalize_embeddings": True}

bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={"device": get_device()},
    encode_kwargs=encode_kwargs
)
