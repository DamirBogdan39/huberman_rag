import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_id = 'naver/splade-cocondenser-ensembledistil'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)


def embed_splade(text: str) -> torch.Tensor:
    """
    A function to perform splade embeddings on a text.

    Parameter
    ----------
    text: str
        A string that needs to be embedded.

    Returns
    ----------
    vec: torch.Tensor
        An embeddings vector of the text.
    """
    tokens = tokenizer(text, return_tensors="pt")
    output = model(**tokens)
    vec = torch.max(
        torch.log(
            1 + torch.relu(output.logits)
        ) * tokens.attention_mask.unsqueeze(-1),
        dim=1)[0].squeeze()

    return vec
