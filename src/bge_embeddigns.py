from typing import List

from langchain_huggingface import HuggingFaceEmbeddings

from config.models import models
from src.device import get_device

model_name = models.bge

# True - computes cosine similarity
encode_kwargs = {"normalize_embeddings": True}

bge_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": get_device()},
    encode_kwargs=encode_kwargs,
)


def embed_bge(text: str) -> List[float]:
    """
    A function to perform bge embeddings on a text.

    Parameter
    ----------
    text: str
        A string that needs to be embedded.

    Returns
    ----------
    vec: List[float]
    An embeddings vector of the text.
    """
    vec = bge_embeddings.embed_query(text)
    return vec
