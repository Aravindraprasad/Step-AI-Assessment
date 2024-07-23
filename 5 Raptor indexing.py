def recursive_raptor(embeddings, texts, depth=0, max_depth=3, min_cluster_size=5):
    if depth >= max_depth or len(embeddings) <= min_cluster_size:
        return {"summary": summarize_cluster(texts), "embeddings": embeddings, "texts": texts}

    cluster_probs, gmm_model = gmm_clustering(embeddings)

    clusters = {}
    for cluster_id in set(cluster_probs):
        cluster_indices = [i for i, prob in enumerate(cluster_probs) if prob == cluster_id]
        cluster_embeddings = embeddings[cluster_indices]
        cluster_texts = [texts[i] for i in cluster_indices]

        clusters[cluster_id] = recursive_raptor(cluster_embeddings, cluster_texts, depth + 1, max_depth, min_cluster_size)

    return clusters


raptor_index = recursive_raptor(summary_embeddings, all_chunks)
