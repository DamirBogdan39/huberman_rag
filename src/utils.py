import re
from langchain_core.documents import Document
from typing import List
import pickle
def reformat_text(text: str) -> str:
    """
    Reformats the the text to remove newline if any character except . is before the newline

    Parameter
    ----------
    text: str
        Text that needs to be reformatted.

    Returns
    ----------
    reformatted_text: str
        Reformatted text.
    """
    reformatted_text = re.sub(r"(?<!\.)\n", " ", text)
    return reformatted_text


def get_document_from_pkl(path: str) -> List[Document]:
    """
    Open the pkl binary file to get the documents.

    Parameter
    ----------
    path: str
        Path to the pickled Documents.

    Returns
    ----------
    documents: List[Document]
        List of Documents.
    """
    with open(path, "rb") as f:
        documents = pickle.load(f)

    return documents