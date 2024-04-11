from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List


class MilvusMultiVectorRetriever(BaseRetriever):

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        from pymilvus import Collection, AnnSearchRequest, WeightedRanker, connections
        from src.bge_embeddigns import embed_bge
        from src.splade_embeddings import embed_splade
        host = "127.0.0.1"  # Milvus server host
        port = "19530"  # Milvus server port

        # Connect to Milvus server
        connections.connect(host=host, port=port)
        collection = Collection(name="huberman_rag")

        def query_collection(s: str):
            bge = embed_bge(s)
            splade = embed_splade(s)

            res = collection.hybrid_search(
                reqs=[
                    AnnSearchRequest(
                        data=[bge],  # Replace with your text vector data
                        anns_field='bge_embeddings',  # Textual data vector field
                        param={"metric_type": "IP", "params": {
                            "nprobe": 10}},  # Search parameters
                        limit=2
                    ),
                    AnnSearchRequest(
                        data=[splade],  # Replace with your image vector data
                        anns_field='splade_embeddings',  # Image data vector field
                        param={"metric_type": "IP", "params": {
                            "nprobe": 10}},  # Search parameters
                        limit=2
                    )
                ],
                rerank=WeightedRanker(0.5, 0.5),

                limit=10,
                output_fields=["pk", "page_content"],
            )

            return res
        res = query_collection(query)
        return [[Document(page_content=i.page_content) for i in res[0]]]
