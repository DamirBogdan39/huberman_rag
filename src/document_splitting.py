from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
    is_separator_regex=False,
)


def split_docs(doc: Document) -> list[Document]:
    """
    Takes in a large lanchain Document and splits it using RecursiveCharacterTextSplitter.


    Parameter
    ----------
    doc: Document
        A lanchain Document object that will be splitted.

    Returns
    ----------
    docs: list[Document]
        A list of smaller lanchain Document objects.
    """
    docs = text_splitter.split_documents(doc)
    return docs
