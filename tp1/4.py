import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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

print(f"\nAcurácia no treino: {train_acc:.4f}")
print(f"Acurácia na validação: {val_acc:.4f}")