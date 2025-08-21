import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Carregar dataset
df = pd.read_csv("https://raw.githubusercontent.com/professortiagoinfnet/inteligencia_artificial/main/heart.csv")

# Exibir primeiras linhas
# print("Visualização inicial dos dados:")
# print(df.head())

# Identificar variável alvo e features
target = "HeartDisease"
features = df.columns.drop(target)

print("\nTarget:", target)
print("\nFeatures:", features)

# Separar variáveis independentes (X) e dependente (y)
X = df[features]
y = df[target]

# Dividir em treino (80%) e validação (20%), mantendo a proporção das classes
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTamanho do conjunto de treino: {len(X_train)} registros")
print(f"Tamanho do conjunto de validação: {len(X_val)} registros")

# Identificar colunas numéricas e categóricas
num_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
cat_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Criar transformador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),       # Padronização para numéricas
        ('cat', OneHotEncoder(drop='first'), cat_features) # Codificação para categóricas
    ]
)

# Ajustar e transformar os conjuntos de treino e validação
X_train_transformed = preprocessor.fit_transform(X_train)
X_val_transformed = preprocessor.transform(X_val)

print("\nFormato após transformação - Treino:", X_train_transformed.shape)
print("Formato após transformação - Validação:", X_val_transformed.shape)

# Criar e treinar o modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)  # pode testar outros valores
knn.fit(X_train_transformed, y_train)

# Fazer previsões
y_train_pred = knn.predict(X_train_transformed)
y_val_pred = knn.predict(X_val_transformed)

# Avaliar acurácia
train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)

# Acurácia
print(f"\nAcurácia no treino: {train_acc:.4f}")
print(f"Acurácia na validação: {val_acc:.4f}")

# Matriz de confusão
cm = confusion_matrix(y_val, y_val_pred)
print("\nMatriz de Confusão (Validação):")
print(cm)

# Relatório de classificação
print("\nRelatório de Classificação (Validação):")
print(classification_report(y_val, y_val_pred))

k_values = range(1, 21)  # Testar K de 1 até 20
train_accuracies = []
val_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_transformed, y_train)
    
    # Acurácia no treino e validação
    train_accuracies.append(accuracy_score(y_train, knn.predict(X_train_transformed)))
    val_accuracies.append(accuracy_score(y_val, knn.predict(X_val_transformed)))

# Mostrar resultados numéricos
print("\nK | Acurácia Treino | Acurácia Validação")
for k, acc_train, acc_val in zip(k_values, train_accuracies, val_accuracies):
    print(f"{k:2} | {acc_train:.4f}        | {acc_val:.4f}")

# Plotar gráfico
plt.figure(figsize=(8,5))
plt.plot(k_values, train_accuracies, marker='o', label='Treino')
plt.plot(k_values, val_accuracies, marker='s', label='Validação')
plt.xlabel("Número de Vizinhos (K)")
plt.ylabel("Acurácia")
plt.title("Impacto de K na Acurácia do KNN")
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.show()