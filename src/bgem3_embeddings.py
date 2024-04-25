from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from src.device import get_device

ef = BGEM3EmbeddingFunction(use_fp16=False, device=get_device())

def embed_bgem3(text:str):
    """
    A function to get the sparse and dense embeddigns from BGEM3EmbeddingFunction.

    Parameter
    ----------
    text: str
        A string that needs to be embedded.

    Returns
    ----------
    dense_emb: List[float]
        A dense embedding vector of the text.

    sparse_emb: 
        A sparse embedding vector of the text.
    """
    embeddings = ef([text])
    dense_embeddings = embeddings["dense"]
    sparse_embeddings = embeddings["sparse"]

    return dense_embeddings, sparse_embeddings