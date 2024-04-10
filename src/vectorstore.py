from langchain.schema.document import Document
from src.splade_embeddings import embed_splade
from src.bge_embeddigns import bge_embeddings
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    MilvusClient
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
    # source = FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500)
    # title = FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500)
    # description = FieldSchema(
    #     name="description", dtype=DataType.VARCHAR, max_length=500)
    # view_count = FieldSchema(name="view_count", dtype=DataType.INT64)
    # thumbnail_url = FieldSchema(
    #     name="thumbnail_url", dtype=DataType.VARCHAR, max_length=500)
    # publish_date = FieldSchema(
    #     name="publish_date", dtype=DataType.VARCHAR, max_length=500)
    # length = FieldSchema(name="length", dtype=DataType.VARCHAR, max_length=500)
    # author = FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=500)

    # Define collection schema
    schema = CollectionSchema(
        fields=[pk, bge_embeddings, splade_embeddings, page_content])
    #   source, title, description,
    #   view_count, thumbnail_url, publish_date, length, author])

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
    # source = doc.metadata["source"]
    # title = doc.metadata["title"]
    # description = doc.metadata["description"]
    # view_count = doc.metadata["view_count"]
    # thumbnail_url = doc.metadata["thumbnail_url"]
    # publish_date = doc.metadata["publish_date"]
    # length = doc.metadata["length"]
    # author = doc.metadata["author"]

    splade_emb = embed_splade(page_content)
    bge_emb = bge_embeddings.embed_query(page_content)

    json_doc = {
        "page_content": page_content,
        # "source": source,
        # "title": title,
        # "description": description,
        # "view_count": view_count,
        # "thumbnail_url": thumbnail_url,
        # "publish_date": publish_date,
        # "length": length,
        # "author": author,
        "splade_embeddings": splade_emb,
        "bge_embeddings": bge_emb
    }
    return json_doc
