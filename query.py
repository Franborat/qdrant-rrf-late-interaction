from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from qdrant_client import QdrantClient, models

if __name__ == '__main__':

    # Initialize Qdrant client (local instance)
    client = QdrantClient(
        url="http://localhost:6333"
    )

    # Define the collection name where documents are stored
    collection_name = "demo_collection"

    # Define the search query
    query = "My query"

    # User ID (optional, can be used for filtering results)
    user_id = ""

    # Load pre-trained models for dense, sparse, and late interaction embeddings
    dense_embedding_model = TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')
    sparse_embedding_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

    # Generate dense, sparse, and late interaction embeddings for the query
    dense_embeddings = list(dense_embedding_model.embed(query))[0]  # Convert generator to list and get first vector
    sparse_embedding_result = list(sparse_embedding_model.embed(query))[0]  # Same for sparse
    late_interaction_embeddings = list(late_interaction_embedding_model.embed(doc for doc in query))[0]

    # Define the prefetching strategy using Reciprocal Rank Fusion (RRF)
    sparse_dense_rrf_prefetch = models.Prefetch(
        prefetch=[
            models.Prefetch(
                query=models.SparseVector(indices=sparse_embedding_result.indices, values=sparse_embedding_result.values),  # <-- sparse vector
                using="splade",
                limit=20,
                # filter=models.Filter(
                #     must=[
                #         models.FieldCondition(
                #             key="user_id",
                #             match=models.MatchValue(value="1"),
                #         )
                #     ]
                # )
            ),
            models.Prefetch(
                query=dense_embeddings,  # <-- dense vector
                using="dense",
                limit=20,
                # filter=models.Filter(
                #     must=[
                #         models.FieldCondition(
                #             key="user_id",
                #             match=models.MatchValue(value="1"),
                #         )
                #     ]
                # )
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
    )

    # Execute the query with late interaction reranking
    response = client.query_points(
        collection_name=collection_name,
        prefetch=sparse_dense_rrf_prefetch,
        query=late_interaction_embeddings,
        using="late",
        with_payload=True,
        limit=10
    )

    print(response)
