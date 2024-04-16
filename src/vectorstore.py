from langchain.schema.document import Document
from src.splade_embeddings import embed_splade
from src.bge_embeddigns import embed_bge
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


def create_collection():
    """
    A function that creates the collection. 
    Connects to milvus and creates fields, collection and indicies for vector field.
    """
    print("Creating collection...")
    client = MilvusClient(
        uri="http://localhost:19530"
    )
    # Specify Milvus server parameters
    host = "127.0.0.1"  # Milvus server host
    port = "19530"  # Milvus server port

    # Connect to Milvus server
    connections.connect(host=host, port=port)
    # Define field schemas
    pk = FieldSchema(name="pk", dtype=DataType.INT64,
                     is_primary=True, auto_id=False)
    bge_embeddings = FieldSchema(
        name="bge_embeddings", dtype=DataType.FLOAT_VECTOR, dim=1024)
    splade_embeddings = FieldSchema(
        name="splade_embeddings", dtype=DataType.FLOAT_VECTOR, dim=30522)
    page_content = FieldSchema(
        name="page_content", dtype=DataType.VARCHAR, max_length=3000)

    # Define collection schema
    schema = CollectionSchema(
        fields=[pk, bge_embeddings, splade_embeddings, page_content])

    # Create collection
    collection_name = "huberman_rag"
    collection = Collection(name=collection_name, schema=schema,)

    index_params = MilvusClient.prepare_index_params()

    index_params.add_index(
        field_name="bge_embeddings",
        metric_type="IP",
        index_type="HNSW",
        index_name="bge_embeddings_index",
        efConstruction=500,
        M=2048
    )

    index_params.add_index(
        field_name="splade_embeddings",
        metric_type="IP",
        index_type="HNSW",
        index_name="splade_embeddings_index",
        efConstruction=500,
        M=2048
    )

    client.create_index(
        collection_name="huberman_rag",
        index_params=index_params
    )
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
    bge = embed_bge(query)
    splade = embed_splade(query)
    collection = activate_collection()
    collection.load()
    res = collection.hybrid_search(
        reqs=[
            AnnSearchRequest(
                data=[bge],
                anns_field="bge_embeddings",
                param={"metric_type": "IP",
                       "params": {"nprobe": 10}},
                limit=5
            ),
            AnnSearchRequest(
                data=[splade],
                anns_field="splade_embeddings",
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
    bge = embed_bge(query)
    collection = activate_collection()
    collection.load()
    res = collection.search(
        data=[bge],
        anns_field="bge_embeddings",
        param={"metric_type": "IP",
               "params": {"nprobe": 10}},
        limit=100,
        output_fields=["page_content"]
    )

    return res
