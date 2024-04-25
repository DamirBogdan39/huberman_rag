from langchain.schema.document import Document
from src.splade_embeddings import embed_splade
from src.bge_embeddigns import embed_bge
from src.bgem3_embeddings import embed_bgem3
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    MilvusClient,
    AnnSearchRequest,
    WeightedRanker
)
import cohere
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")


def create_collection():
    """
    A function that creates the collection. 
    Connects to milvus and creates fields, collection and indicies for vector field.
    """
    print("Creating collection...")
    connections.connect("default", host="localhost", port="19530")

    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR,
                    is_primary=True, auto_id=True, max_length=100),
        FieldSchema(name="page_content", dtype=DataType.VARCHAR, max_length=3000),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
    ]

    schema= CollectionSchema(fields, "")

    col = Collection("huberman_rag", schema)

    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP", "efConstruction": 500, "M": 2048}
    dense_index = {"index_type": "HNSW", "metric_type": "IP", "efConstruction": 500, "M": 2048}

    col.create_index("sparse_vector", sparse_index)
    col.create_index("dense_vector", dense_index)
    print("Collection created successfully.")


def jsonize_document(doc: Document) -> dict:
    page_content = doc.page_content
    splade_emb = embed_splade(page_content)
    bge_emb = embed_bge(page_content)

    json_doc = {
        "page_content": page_content,
        "splade_embeddings": splade_emb,
        "bge_embeddings": bge_emb
    }
    return json_doc


def connect_to_milvus():
    host = "127.0.0.1"  # Milvus server host
    port = "19530"  # Milvus server port

    # Connect to Milvus server
    connections.connect(host=host, port=port)


def activate_collection():
    collection = Collection(name="huberman_rag")
    return collection


def multivector_query(query: str) -> str:
    """

    """
    dense_embeddings = embed_bge(query)
    sparse_embeddings = embed_splade(query)
    collection = activate_collection()
    collection.load()
    res = collection.hybrid_search(
        reqs=[
            AnnSearchRequest(
                data=[dense_embeddings],
                anns_field="dense_vector",
                param={"metric_type": "IP",
                       "params": {"nprobe": 10}},
                limit=5
            ),
            AnnSearchRequest(
                data=sparse_embeddings,
                anns_field="sparse_vector",
                param={"metric_type": "IP",
                       "params": {"nprobe": 10}},
                limit=5
            )
        ],
        rerank=WeightedRanker(0.8, 0.2),

        limit=10,
        output_fields=["page_content"],
    )

    return res


def rerank_query(query: str) -> str:
    """

    """
    dense_embeddings = embed_bge(query)
    collection = activate_collection()
    collection.load()
    unranked_results = collection.search(
        data=[dense_embeddings],
        anns_field="dense_vector",
        param={"metric_type": "IP",
               "params": {"nprobe": 10}},
        limit=100,
        output_fields=["page_content"]
    )

    docs_dict = {i: doc for i, doc in enumerate(unranked_results[0])}
    co = cohere.Client(COHERE_API_KEY)
    reranked_docs = co.rerank(query=query,
                              documents=[
                                  doc.page_content for doc in docs_dict.values()],
                              top_n=10,
                              model="rerank-english-v2.0")
    indices = [response.index for response in reranked_docs.results]
    reranked_docs = [Document(unranked_results[0][i].page_content)
                     for i in indices]
    return reranked_docs
