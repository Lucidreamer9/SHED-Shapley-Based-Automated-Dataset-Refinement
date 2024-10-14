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
# # 1. For each cluster, compute the distance of all data points to the cluster center.
# avg_distances = []

# for cluster_id, sentences in enumerate(clustered_sentences):
#     distances = []
#     center_embedding = clustering_model.cluster_centers_[cluster_id]
#     for sentence in sentences:
#         sentence_embedding = embedder.encode([sentence])
#         distance = np.linalg.norm(sentence_embedding - center_embedding)
#         distances.append(distance)
    
#     # 2. Compute the average of these distances.
#     avg_distance = sum(distances) / len(distances) if distances else 0
    
#     # 3. 将结果存储为字典，key是cluster的文件名，value是平均距离。
#     cluster_filename = "cluster"+str(cluster_id)
#     avg_distances.append({cluster_filename: avg_distance})

# # 4. 将所有这些字典添加到一个列表中。
# # 此步骤已经在上面的循环中完成

# # 5. 将该列表存储在txt文件中。
# with open("path/to/avg_distances.txt", 'w', encoding="utf-8") as file:
#     for item in avg_distances:
#         file.write(json.dumps(item) + "\n")
