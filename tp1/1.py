import pandas as pd

# Carregar dataset
df = pd.read_csv("https://raw.githubusercontent.com/professortiagoinfnet/inteligencia_artificial/main/heart.csv")

# Exibir primeiras linhas
print("Visualização inicial dos dados:")
print(df.head())

# Identificar variável alvo e features
target = "HeartDisease"
features = df.columns.drop(target)

print("\nTarget:", target)
print("\nFeatures:", features)
