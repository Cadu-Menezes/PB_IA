
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
#===========================================
# Questão 1 (Clusterização K-Médias)
#===========================================

# CARREGAR O DATASET
df = pd.read_csv("dataset.csv")  # baixar o arquivo em https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset?utm_source=chatgpt.com e coloca-lo na mesma pasta do script
df = df.head(10000)  # Usar apenas as primeiras 10.000 linhas, pois o dataset é muito grande

# PRÉ-PROCESSAMENTO INICIAL
print(df.columns)  # Confirmar colunas visíveis

# Remover colunas não numéricas ou irrelevantes
df_clean = df.drop(columns=[
    'Unnamed: 0', 'track_id', 'artists', 'album_name', 'track_name'
])
print("Colunas após remoção:", df_clean.columns)

# Remover valores ausentes
df_clean.dropna(inplace=True)

# ENCODER DO TARGET (track_genre)
label_encoder = LabelEncoder()
df_clean['genre_encoded'] = label_encoder.fit_transform(df_clean['track_genre'])

# Separar features e target
X = df_clean.drop(columns=['track_genre', 'genre_encoded'])
y = df_clean['genre_encoded']

# DIVISÃO EM TREINO E TESTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# PADRONIZAÇÃO DAS FEATURES
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# K-MEANS CLUSTERING (placeholder)
# Usar o X_train padronizado para clusterização
inertia = []
silhouette_scores = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train_scaled)
    
    inertia.append(kmeans.inertia_)
    silhouette = silhouette_score(X_train_scaled, kmeans.labels_)
    silhouette_scores.append(silhouette)

# Plot Elbow Method
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(k_values, inertia, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia')

# Plot Silhouette Score
plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, marker='s', color='green')
plt.title('Índice de Silhueta')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()

# ==========================================
# Questão 2 (SVM com Feature de Cluster)
# ==========================================
kmeans = KMeans(n_clusters=6, random_state=42)
kmeans.fit(X_train_scaled)

# Para cada ponto, calcular a distância ao centroide mais próximo
X_train_dist = kmeans.transform(X_train_scaled)
X_test_dist = kmeans.transform(X_test_scaled)

# Criar nova feature: distância ao centroide mais próximo
X_train_cluster_feature = np.min(X_train_dist, axis=1).reshape(-1, 1)
X_test_cluster_feature = np.min(X_test_dist, axis=1).reshape(-1, 1)

# Concatenar com os dados padronizados originais
X_train_with_cluster = np.hstack([X_train_scaled, X_train_cluster_feature])
X_test_with_cluster = np.hstack([X_test_scaled, X_test_cluster_feature])
# ==========================================

# SVM SEM FEATURE DE CLUSTER (baseline)
svm_baseline = SVC()
svm_baseline.fit(X_train_scaled, y_train)
y_pred_baseline = svm_baseline.predict(X_test_scaled)

print("=== SVM SEM FEATURE DE CLUSTER ===")
print("Acurácia:", accuracy_score(y_test, y_pred_baseline))
print(classification_report(y_test, y_pred_baseline))

# SVM COM FEATURE DE DISTÂNCIA AO CLUSTER
svm_cluster = SVC()
svm_cluster.fit(X_train_with_cluster, y_train)
y_pred_cluster = svm_cluster.predict(X_test_with_cluster)

print("=== SVM COM FEATURE DE DISTÂNCIA AO CLUSTER ===")
print("Acurácia:", accuracy_score(y_test, y_pred_cluster))
print(classification_report(y_test, y_pred_cluster))

# Comparação dos resultados
print("Baseline Accuracy:", accuracy_score(y_test, y_pred_baseline))
print("Com Cluster Accuracy:", accuracy_score(y_test, y_pred_cluster))

# ==========================================
# 3.a - SVM com diferentes kernels e C
# ==========================================
kernels = ['linear', 'poly', 'rbf']
C_values = [0.1, 1, 10]

print("=== SVM COM FEATURES ORIGINAIS ===")
for kernel in kernels:
    for C in C_values:
        model = SVC(kernel=kernel, C=C)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        print(f"Kernel: {kernel:<6} | C: {C:<4} | Accuracy: {acc:.4f}")

print("\n=== SVM COM FEATURE DE DISTÂNCIA AO CLUSTER ===")
for kernel in kernels:
    for C in C_values:
        model = SVC(kernel=kernel, C=C)
        model.fit(X_train_with_cluster, y_train)
        y_pred = model.predict(X_test_with_cluster)
        acc = accuracy_score(y_test, y_pred)
        print(f"Kernel: {kernel:<6} | C: {C:<4} | Accuracy: {acc:.4f}")

# ==========================================
# 3.b - Random Forest com diferentes parâmetros
# ==========================================
n_estimators_list = [100, 200]
max_depth_list = [10, 20, None]

print("\n=== RANDOM FOREST COM FEATURES ORIGINAIS ===")
for n in n_estimators_list:
    for depth in max_depth_list:
        rf = RandomForestClassifier(n_estimators=n, max_depth=depth, random_state=42)
        rf.fit(X_train_scaled, y_train)
        y_pred = rf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        print(f"n_estimators: {n:<3} | max_depth: {str(depth):<4} | Accuracy: {acc:.4f}")

print("\n=== RANDOM FOREST COM FEATURE DE DISTÂNCIA AO CLUSTER ===")
for n in n_estimators_list:
    for depth in max_depth_list:
        rf = RandomForestClassifier(n_estimators=n, max_depth=depth, random_state=42)
        rf.fit(X_train_with_cluster, y_train)
        y_pred = rf.predict(X_test_with_cluster)
        acc = accuracy_score(y_test, y_pred)
        print(f"n_estimators: {n:<3} | max_depth: {str(depth):<4} | Accuracy: {acc:.4f}")

#===========================================
#Questao 4 - Avaliação com AUC-ROC
#===========================================

# Binarizar os rótulos para cálculo do AUC-ROC multiclasse
y_test_bin = label_binarize(y_test, classes=np.unique(y))

# ========= Avaliar melhor SVM COM feature =========
svm_best = SVC(kernel='rbf', C=10, probability=True)  # probability=True é necessário
svm_best.fit(X_train_with_cluster, y_train)
y_pred = svm_best.predict(X_test_with_cluster)
y_prob = svm_best.predict_proba(X_test_with_cluster)

print("=== MELHOR SVM (COM FEATURE) ===")
print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='macro')
print(f"AUC-ROC (macro): {roc_auc:.4f}")

# ========= Avaliar melhor RF COM feature =========
rf_best = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
rf_best.fit(X_train_with_cluster, y_train)
y_pred_rf = rf_best.predict(X_test_with_cluster)
y_prob_rf = rf_best.predict_proba(X_test_with_cluster)

print("\n=== MELHOR RANDOM FOREST (COM FEATURE) ===")
print(classification_report(y_test, y_pred_rf))
roc_auc_rf = roc_auc_score(y_test_bin, y_prob_rf, multi_class='ovr', average='macro')
print(f"AUC-ROC (macro): {roc_auc_rf:.4f}")


#=============================
#Questao 5
#=============================

labels = ['SVM', 'Random Forest']
acc_sem_cluster = [0.6090, 0.6580]
acc_com_cluster = [0.6145, 0.6535]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, acc_sem_cluster, width, label='Sem Feature de Cluster')
plt.bar(x + width/2, acc_com_cluster, width, label='Com Feature de Cluster')

plt.ylabel('Acurácia')
plt.title('Comparação de Acurácia com e sem Feature de Cluster')
plt.xticks(x, labels)
plt.ylim(0.55, 0.70)
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()