import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# Carregar datasets e unificar
fake = pd.read_csv("https://raw.githubusercontent.com/professortiagoinfnet/inteligencia_artificial/main/Fake.csv")
true = pd.read_csv("https://raw.githubusercontent.com/professortiagoinfnet/inteligencia_artificial/main/True.csv")

fake = fake.rename(columns={fake.columns[0]:"title", fake.columns[1]:"text", fake.columns[2]:"subject", fake.columns[3]:"date"})
true = true.rename(columns={true.columns[0]:"title", true.columns[1]:"text", true.columns[2]:"subject", true.columns[3]:"date"})

fake["label"], fake["target"] = "fake", 0
true["label"], true["target"] = "true", 1

df = pd.concat([fake, true], ignore_index=True)

print(df.head())
print(df["target"].value_counts())

# Definir X e y 
X = df["text"]
y = df["target"]

# Split treino/validação 
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Vetorização TF-IDF - QUESTÃO 1
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=100_000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf   = vectorizer.transform(X_val)

# Treinar KNN (holdout) para diferentes K - QUESTÃO 2 E 2.A
k_values = [1, 3, 5, 7, 9, 11, 15, 20]
results_holdout = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_train_tfidf, y_train)
    y_pred = knn.predict(X_val_tfidf)
    acc = accuracy_score(y_val, y_pred)
    results_holdout.append((k, acc))
    print(f"[HOLDOUT] K={k} -> Acurácia: {acc:.4f}")

# Validação Cruzada com TF-IDF dentro do Pipeline
#      -> evita vazamento: TF-IDF é fitado só no treino de cada fold
# Configurar a validação cruzada - QUESTÃO 3
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results_cv = []

for k in k_values:
    pipe = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=100_000,
            stop_words="english"
        )),
        ("knn", KNeighborsClassifier(n_neighbors=k, n_jobs=-1))
    ])

    scores = cross_val_score(
        pipe, X, y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )
    results_cv.append((k, scores.mean(), scores.std()))
    print(f"[CV 5-fold] K={k} -> Acc média: {scores.mean():.4f} (+/- {scores.std():.4f})")

# 6) Resumos
df_holdout = pd.DataFrame(results_holdout, columns=["K", "Acurácia (holdout)"])\
             .sort_values("Acurácia (holdout)", ascending=False).reset_index(drop=True)
df_cv = pd.DataFrame(results_cv, columns=["K", "Acc média (CV)", "Desvio Padrão (CV)"])\
        .sort_values("Acc média (CV)", ascending=False).reset_index(drop=True)

print("\nResumo (Holdout):")
print(df_holdout)
print("\nResumo (Validação Cruzada 5-fold):")
print(df_cv)
