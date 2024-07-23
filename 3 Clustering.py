from sklearn.mixture import GaussianMixture
import numpy as np

def gmm_clustering(embeddings, n_components=10):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    probabilities = gmm.fit_predict(embeddings)
    return probabilities, gmm

cluster_probs, gmm_model = gmm_clustering(embeddings)