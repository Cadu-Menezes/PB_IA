import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Carregar o dataset
url = "https://raw.githubusercontent.com/cassiusf/datasets/refs/heads/main/titanic_data.csv"
df = pd.read_csv(url)

# Eliminar colunas pedidas
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

# Eliminar registros com dados ausentes
df.dropna(inplace=True)

# Aplicar LabelEncoder nas colunas categóricas
label_encoder = LabelEncoder()
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
df['Sex'] = label_encoder.fit_transform(df['Sex'])

# Separar features e target
X = df.drop(columns=['Survived'])  # Features
y = df['Survived']                 # Target

# Dividir em treino e teste (75% treino, 25% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Treinar modelo SVM com kernel linear
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Avaliação
y_pred = svm_model.predict(X_test)
print("Relatório de Classificação (SVM - Kernel Linear):")
print(classification_report(y_test, y_pred))
