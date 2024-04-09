from src.youtube_loader import load_from_youtube
from src.bge_embeddigns import bge_embeddings
from src.splade_embeddings import embed_splade
from src.document_splitting import text_splitter
import logging

doc = load_from_youtube("https://www.youtube.com/watch?v=tkH2-_jMCSk&t")


# embedding = bge_embeddings.embed_query(doc[0].page_contentp[:1000])
# splades = embed_splade(doc[0].page_content[:1000])
# print(len(embedding), splades.shape)


texts = text_splitter.create_documents([doc[0].page_content])
print(len(texts))
