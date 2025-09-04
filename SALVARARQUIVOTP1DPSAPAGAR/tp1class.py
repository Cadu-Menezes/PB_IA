import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

#==========================================
# Questao1
#==========================================

# a. Carregar o dataset
url = "https://raw.githubusercontent.com/cassiusf/datasets/refs/heads/main/titanic_data.csv"
df = pd.read_csv(url)

# b. Eliminar as variáveis indesejadas
df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

# c. Eliminar observações com dados ausentes (NA)
df.dropna(inplace=True)

# d. Aplicar LabelEncoder nas variáveis categóricas
label_encoder = LabelEncoder()
df["Embarked"] = label_encoder.fit_transform(df["Embarked"])
df["Sex"] = label_encoder.fit_transform(df["Sex"])

# e. Separar em treino e teste (80/20)
df_treino, df_teste = train_test_split(df, test_size=0.2, random_state=42)

# Exibir shapes dos conjuntos
print("Shape do conjunto de treino:", df_treino.shape)
print("Shape do conjunto de teste:", df_teste.shape)


#===================================
#Questao2
#===================================

# Separar features e target
X_train = df_treino.drop(columns=["Survived"])
y_train = df_treino["Survived"]
X_test = df_teste.drop(columns=["Survived"])
y_test = df_teste["Survived"]

# a. Criar e treinar modelo de árvore de decisão com parâmetros default
modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = modelo.predict(X_test)

# a. Matriz de confusão
matriz = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print(matriz)

# b. Valores TN, TP, FN, FP
tn, fp, fn, tp = matriz.ravel()
print(f"TN (Verdadeiros Negativos): {tn}")
print(f"TP (Verdadeiros Positivos): {tp}")
print(f"FN (Falsos Negativos): {fn}")
print(f"FP (Falsos Positivos): {fp}")

# c. Acurácia
acuracia = accuracy_score(y_test, y_pred)
print(f"Acurácia: {acuracia:.4f}")

# d. Precision, Recall, F1-score
relatorio = classification_report(y_test, y_pred, digits=4)
print("Relatório de Classificação:")
print(relatorio)


#=============================
#Questao3
#=============================

def avaliar_modelo_arvore(max_depth):
    print(f"\n### Avaliando árvore com max_depth={max_depth} ###")
    modelo = DecisionTreeClassifier(max_depth=max_depth)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    matriz = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = matriz.ravel()
    acuracia = accuracy_score(y_test, y_pred)
    relatorio = classification_report(y_test, y_pred, digits=4)

    print("Matriz de Confusão:")
    print(matriz)
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    print(f"Acurácia: {acuracia:.4f}")
    print("Relatório de Classificação:")
    print(relatorio)


# Avaliar para max_depth = 3 e 5
avaliar_modelo_arvore(3)
avaliar_modelo_arvore(5)

#=======================
#Questao4
#=======================

# Re-treinar os modelos só para garantir 
modelo_default = DecisionTreeClassifier()
modelo_default.fit(X_train, y_train)

modelo_depth3 = DecisionTreeClassifier(max_depth=3)
modelo_depth3.fit(X_train, y_train)

modelo_depth5 = DecisionTreeClassifier(max_depth=5)
modelo_depth5.fit(X_train, y_train)

# Função para exibir a árvore
def visualizar_arvore(modelo, titulo):
    plt.figure(figsize=(20, 10))
    plot_tree(modelo, feature_names=X_train.columns, class_names=["Not Survived", "Survived"], filled=True)
    plt.title(titulo)
    plt.show()

# Visualizações
visualizar_arvore(modelo_default, "Árvore de Decisão - Modelo Original (Default)")
visualizar_arvore(modelo_depth3, "Árvore de Decisão - max_depth=3")
visualizar_arvore(modelo_depth5, "Árvore de Decisão - max_depth=5")