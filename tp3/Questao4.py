# ==============================================
# QUESTÃO 4 - QUANTIZAÇÃO VETORIAL COM K-MEANS
# ==============================================

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 1) Gerar dataset sintético com 4 grupos
X, _ = make_blobs(n_samples=1000, centers=4, cluster_std=1.2, random_state=42)

# 2) Aplicar K-Means para criar 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X)

# Centroides aprendidos pelo algoritmo
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# 3) Plotar os dados originais e os centroides
plt.figure(figsize=(10, 5))

# Dados coloridos pelos clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=15)
plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="X", s=200, label="Centroides")

plt.title("Quantização Vetorial com K-Means")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# 4) Substituir os pontos pelos centroides -> Quantização
X_quantized = centroids[labels]

# 5) Plotar os dados após quantização
plt.figure(figsize=(10, 5))
plt.scatter(X_quantized[:, 0], X_quantized[:, 1], c=labels, cmap="viridis", s=15)
plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="X", s=200, label="Centroides")

plt.title("Dados Após Quantização Vetorial")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()