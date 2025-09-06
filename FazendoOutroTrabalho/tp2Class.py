import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC

#============
#Questao 1
#============

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

#============
#Questao 2
#============

# a) Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()

# b) Métricas de desempenho
acuracia = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Acurácia: {acuracia:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

#============
#Questao 3
#============

# Função para treinar e avaliar um modelo SVM com kernel específico
def treinar_avaliar_svm(kernel_name):
    print(f"\nAvaliação com kernel: {kernel_name.upper()}")
    model = SVC(kernel=kernel_name)
    model.fit(X_train, y_train)
    y_pred_kernel = model.predict(X_test)

    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred_kernel)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title(f"Matriz de Confusão - Kernel {kernel_name.upper()}")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.show()

    # Métricas
    acuracia = accuracy_score(y_test, y_pred_kernel)
    precision = precision_score(y_test, y_pred_kernel)
    recall = recall_score(y_test, y_pred_kernel)
    f1 = f1_score(y_test, y_pred_kernel)

    print(f"Acurácia: {acuracia:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")


treinar_avaliar_svm('rbf')
treinar_avaliar_svm('sigmoid')