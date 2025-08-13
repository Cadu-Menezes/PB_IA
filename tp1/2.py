import pandas as pd
from sklearn.model_selection import train_test_split

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