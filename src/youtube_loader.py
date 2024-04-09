from langchain_community.document_loaders import YoutubeLoader
from langchain.schema.document import Document


def load_from_youtube(url: str) -> Document:
    """
    To do.
    """
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    doc = loader.load()
    return doc
