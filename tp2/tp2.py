import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# =========================================
# 1) Carregar datasets e unificar
# =========================================
# Carregar datasets
fake = pd.read_csv("https://raw.githubusercontent.com/professortiagoinfnet/inteligencia_artificial/main/Fake.csv")
true = pd.read_csv("https://raw.githubusercontent.com/professortiagoinfnet/inteligencia_artificial/main/True.csv")

# Renomear colunas do Fake
fake = fake.rename(columns={
    fake.columns[0]: "title",
    fake.columns[1]: "text",
    fake.columns[2]: "subject",
    fake.columns[3]: "date"
})
fake["label"] = "fake"
fake["target"] = 0  # Fake = 0

# Renomear colunas do True
true = true.rename(columns={
    true.columns[0]: "title",
    true.columns[1]: "text",
    true.columns[2]: "subject",
    true.columns[3]: "date"
})
true["label"] = "true"
true["target"] = 1  # True = 1

# Concatenar datasets
df = pd.concat([fake, true], ignore_index=True)

# Mostrar informações do dataset final
print(df.head())
print(df["target"].value_counts())


# =========================================
# 3) Split treino/validação
# =========================================
X = df["text"]
y = df["target"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================
# 4) Vetorização TF-IDF
# =========================================
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=100_000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf   = vectorizer.transform(X_val)

# =========================================
# 5) Treinar KNN para diferentes valores de K
# =========================================
k_values = [1, 3, 5, 7, 9, 11, 15, 20]
results = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_train_tfidf, y_train)
    y_pred = knn.predict(X_val_tfidf)
    acc = accuracy_score(y_val, y_pred)
    results.append((k, acc))
    print(f"K={k} -> Acurácia: {acc:.4f}")

# =========================================
# 6) Mostrar resultados organizados
# =========================================
df_results = pd.DataFrame(results, columns=["K", "Acurácia"])
print("\nResumo dos resultados:")
print(df_results)
