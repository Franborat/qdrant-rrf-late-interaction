import json
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from qdrant_client import QdrantClient, models

if __name__ == '__main__':

    # Initialize Qdrant client (local instance)
    client = QdrantClient(
        url="http://localhost:6333"
    )

    # Name of the collection where embeddings will be stored
    collection_name = "demo_collection"

    # Path to the JSON file containing document titles and metadata
    payload_path = "titles_with_users.json"

    metadata = [] # List to store metadata associated with documents
    documents = [] # List to store document titles

    # Read the JSON file line by line
    with open(payload_path) as fd:
        for line in fd:
            obj = json.loads(line) # Parse each line as a JSON object

            # Only add documents that have a non-null title
            if obj.get("title") is not None:
                documents.append(obj.pop("title")) # Extract title and add it to documents
                metadata.append(obj) # Store the remaining metadata

    # Select only the first 1000 documents for embedding
    documents_small = documents[0:1000]

    # Load embedding models for dense, sparse, and late interaction embeddings
    dense_embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    sparse_embedding_model = SparseTextEmbedding("prithivida/Splade_PP_en_v1")
    late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

    # Convert our documents into embeddings
    dense_embeddings = list(dense_embedding_model.embed(doc for doc in documents_small))
    sparse_embeddings = list(sparse_embedding_model.embed(doc for doc in documents_small))
    late_interaction_embeddings = list(late_interaction_embedding_model.embed(doc for doc in documents_small))

    # Check if the collection exists; if not, create it
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            shard_number=3, # Set number of shards
            replication_factor=2, # Set replication factor
            vectors_config={
                "dense": models.VectorParams(
                    size=len(dense_embeddings[0]),  # Set vector size based on model output
                    distance=models.Distance.COSINE, # Use cosine similarity for distance computation
                ),
                "late": models.VectorParams(
                    size=len(late_interaction_embeddings[0][0]),  # Late interaction vector size
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM, # Use max similarity for comparison
                    )
                ),
            },
            sparse_vectors_config={"splade": models.SparseVectorParams()}, # Sparse vector configuration
            quantization_config=models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(always_ram=True), # Enable binary quantization in RAM
            ),
            optimizers_config=models.OptimizersConfigDiff(
                default_segment_number=5,  # Number of segments for optimization
                indexing_threshold=0,   # Disable indexing temporarily to speed up data upload
            )
        )

    # Prepare data points to upload
    points = []
    for idx, (dense_embedding, sparse_embedding, late_interaction_embedding, meta) in enumerate(
            zip(dense_embeddings, sparse_embeddings, late_interaction_embeddings, metadata)):

        # Define a Qdrant point with dense, sparse, and late interaction embeddings
        point = models.PointStruct(
            id=idx, # Unique ID for each document
            vector={
                "dense": dense_embedding,
                "splade": models.SparseVector(indices=sparse_embedding.indices, values=sparse_embedding.values),
                "late": late_interaction_embedding,
            },
            payload={"document": meta} # Attach metadata as payload
        )
        points.append(point)

    # Upload all the points (documents and embeddings) to Qdrant
    client.upload_points(collection_name=collection_name, points=points)

    # Now that data is loaded, re-enable indexing for better performance
    client.update_collection(
        collection_name=collection_name,
        optimizer_config=models.OptimizersConfigDiff(
            indexing_threshold=20000
        )
    )
