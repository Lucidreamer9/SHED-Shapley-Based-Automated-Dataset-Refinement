import os
import numpy as np
os.environ["OPENBLAS_NUM_THREADS"] = "64"
os.environ["n_jobs"] = "64"
os.environ["OMP_NUM_THREADS"] = "64"

import json
import sys
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
def main():
    data_path = str(sys.argv[1])
    num_clusters = int(sys.argv[2])
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    corpus=[]

    f=open(data_path)
    jsonObect=json.load(f)
    for i in jsonObect:
        corpus.append(str(i))
    # Load corpus with example sentences

    corpus_embeddings = embedder.encode(corpus)

    # Perform kmean clustering
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(corpus[sentence_id])

    for i, cluster in enumerate(clustered_sentences):
        print("Cluster ", i+1)
        file = open("/workspace/cluster_"+str(num_clusters)+"_"+str(i)+".txt",'a',encoding="utf-8")
        for j in range(len(cluster)):
            file.write(cluster[j]+"\n")


    closest, _ = pairwise_distances_argmin_min(clustering_model.cluster_centers_, corpus_embeddings)
    file = open("/workspace/cluster_center_"+str(num_clusters)+".txt", 'a',encoding="utf-8")
    for j in range(len(closest)):
        file.write(corpus[closest[j]]+"\n")

if __name__ == "__main__":
    main()

