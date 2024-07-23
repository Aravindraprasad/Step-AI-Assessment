import numpy as np

def process_cluster(cluster):
    if isinstance(cluster, dict) and 'embeddings' in cluster:
        embeddings = cluster['embeddings']
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        
        return [{
            "embedding": embedding,
            "summary": cluster.get('summary', ''),
            "text": cluster.get('text', '')
        } for embedding in embeddings]
    elif isinstance(cluster, str):
        print(f"Found string data: {cluster[:50]}...")
        return []
    elif isinstance(cluster, np.ndarray):
        print(f"Found numpy array of shape: {cluster.shape}")
        return [{
            "embedding": embedding.tolist(),
            "summary": "",
            "text": ""
        } for embedding in cluster]
    else:
        print(f"Unexpected structure: {type(cluster)}")
        return []

entities = []

def process_nested_structure(structure):
    if isinstance(structure, dict):
        for value in structure.values():
            process_nested_structure(value)
    elif isinstance(structure, (list, tuple)):
        for item in structure:
            process_nested_structure(item)
    else:
        entities.extend(process_cluster(structure))

process_nested_structure(raptor_index)

if entities:
    try:
        collection.insert(entities)
        print(f"Inserted {len(entities)} entities into the collection")
    except Exception as e:
        print(f"Error inserting entities: {e}")
        print("First few entities:", entities[:3])
else:
    print("No valid entities found to insert")