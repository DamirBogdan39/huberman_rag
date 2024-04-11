from src.vectorstore import connect_to_milvus, multivector_query

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List


class MilvusMultiVectorRetriever(BaseRetriever):

    def _get_relevant_documents(self,
                                query: str,
                                *,
                                run_manager: CallbackManagerForRetrieverRun) -> List[Document]:

        connect_to_milvus()
        res = multivector_query(query)

        return [[Document(page_content=i.page_content) for i in res[0]]]
