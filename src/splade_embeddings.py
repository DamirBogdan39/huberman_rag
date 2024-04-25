import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from typing import List
from scipy import sparse
import numpy as np

model_id = 'naver/splade-cocondenser-ensembledistil'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)


def embed_splade(text: str):
    """
    A function to perform splade embeddings on a text.

    Parameter
    ----------
    text: str
        A string that needs to be embedded.

    Returns
    ----------
    sparse_arr: List[float]
        A scipy sparse array.
    """
    tokens = tokenizer(text, return_tensors="pt")
    output = model(**tokens)
    vec = torch.max(
        torch.log(
            1 + torch.relu(output.logits)
        ) * tokens.attention_mask.unsqueeze(-1),
        dim=1)[0].squeeze()
    vec = vec.detach().tolist()

    numpy_arr = np.array(vec)
    sparse_arr = sparse.csr_matrix(numpy_arr)

    return sparse_arr
