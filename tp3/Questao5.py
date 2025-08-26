# ==============================================
# QUESTÃO 5 - PCA + K-MEANS SHOPMANIA
# ==============================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import kagglehub
import os

# ==============================================
# 1) Carregar o dataset
# ==============================================
path = kagglehub.dataset_download("lakritidis/product-classification-and-categorization")
print("Arquivo baixado em:", path)

df = pd.read_csv(os.path.join(path, "shopmania.csv"), header=None)
df.columns = ["id", "title", "category_id", "category_name"]

print("Dimensão original do dataset:", df.shape)

# ==============================================
# 1.1) Subamostrar para acelerar o processamento (60k registros)
# ==============================================
df = df.sample(n=60000, random_state=42)
print("Dimensão após subamostragem:", df.shape)

# ==============================================
# 2) Selecionar colunas relevantes
# ==============================================
df = df.dropna(subset=['title'])  # Remove linhas sem título
X_text = df['title'].astype(str)

# ==============================================
# 3) Vetorização TF-IDF (mantendo os mesmos parâmetros)
# ==============================================
tfidf = TfidfVectorizer(max_features=10000, stop_words="english")
X_tfidf = tfidf.fit_transform(X_text)
print("Dimensão do TF-IDF:", X_tfidf.shape)

# ==============================================
# 4) Reduzir dimensionalidade com PCA
# ==============================================
# Vamos reduzir para 100 componentes para equilibrar qualidade e desempenho
pca = PCA(n_components=100, random_state=42)
X_reduced = pca.fit_transform(X_tfidf.toarray())

print("Dimensão após PCA:", X_reduced.shape)

# ==============================================
# 5) Rodar KMeans para diferentes valores de K
# ==============================================
k_values = [5, 10, 15, 20]
results = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_reduced)
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_reduced, labels)
    results.append((k, inertia, silhouette))
    print(f"K={k} -> Inércia={inertia:.2f} | Silhouette={silhouette:.4f}")

df_results = pd.DataFrame(results, columns=["K", "Inércia", "Silhouette"])
print("\nResumo dos resultados com PCA:")
print(df_results)

# ==============================================
# 6) Visualizar a variação da Inércia e Silhouette
# ==============================================
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Elbow Method
sns.lineplot(data=df_results, x="K", y="Inércia", marker="o", ax=ax[0])
ax[0].set_title("Elbow Method - Inércia (com PCA)")
ax[0].set_xlabel("Número de Clusters (K)")
ax[0].set_ylabel("Inércia")

# Silhouette Score
sns.lineplot(data=df_results, x="K", y="Silhouette", marker="o", color="green", ax=ax[1])
ax[1].set_title("Silhouette Score (com PCA)")
ax[1].set_xlabel("Número de Clusters (K)")
ax[1].set_ylabel("Score")

plt.tight_layout()
plt.show()

# ==============================================
# 7) Visualizar clusters em 2D usando PCA
# ==============================================
pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X_tfidf.toarray())

# Usar o melhor K encontrado (ex: 10)
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_reduced)

plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=10)
plt.title("Visualização dos Clusters (PCA 2D)")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.colorbar(label="Cluster")
plt.show()
