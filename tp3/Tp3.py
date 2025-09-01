import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    roc_curve, auc, precision_score, recall_score,
    f1_score, confusion_matrix, RocCurveDisplay
)
from sklearn.preprocessing import LabelBinarizer

#==========================================
#1) Criação das features:Aplicar Análise de Componentes Principais (PCA) para reduzir a dimensionalidade dos conjuntos de dados.
#==========================================
# Carregar o dataset sem cabeçalho
url = "https://raw.githubusercontent.com/professortiagoinfnet/inteligencia_artificial/main/sonar_dataset.csv"

# Criar nomes das colunas: 60 colunas + 1 de classe
col_names = [f'feature_{i}' for i in range(60)] + ['Class']
df = pd.read_csv(url, header=None, names=col_names)

# Separar features (X) e classe (y)
X = df.drop(columns=['Class'])
y = df['Class']

# Padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA (mantendo todas as componentes)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Mostrar shapes
print("Shape original:", X.shape)
print("Shape após PCA:", X_pca.shape)

# Mostrar variância explicada
plt.figure(figsize=(8,5))
plt.plot(range(1, 61), pca.explained_variance_ratio_.cumsum(), marker='o')
plt.title("Variância Acumulada por Componentes Principais")
plt.xlabel("Número de Componentes")
plt.ylabel("Variância Explicada Acumulada")
plt.grid()
plt.show()


#==========================================
#2) Modelo de ML:  Desenvolver e treinar modelos de árvores de decisão para tarefas de classificação.
#==========================================

# Dividir os dados transformados com PCA em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)

# Criar e treinar o modelo de árvore de decisão
modelo_arvore = DecisionTreeClassifier(random_state=42)
modelo_arvore.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = modelo_arvore.predict(X_test)

# Avaliar o modelo
print("\n Avaliação do Modelo de Árvore de Decisão:")
print("Acurácia:", accuracy_score(y_test, y_pred))

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

#==========================================
#3) 3) Avaliação de Modelos: Aplicar técnicas de validação cruzada para estimar a eficiência dos modelos desenvolvidos.
#==========================================

# Instanciar o validador estratificado com 5 folds
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Reutilizar o mesmo modelo (árvore de decisão)
modelo_cv = DecisionTreeClassifier(random_state=42)

# Executar a validação cruzada
scores = cross_val_score(modelo_cv, X_pca, y, cv=kfold, scoring='accuracy')

# Exibir os resultados
print("\n Validação Cruzada (5 Folds)")
print("Scores por fold:", scores)
print("Acurácia média: {:.4f}".format(scores.mean()))
print("Desvio padrão: {:.4f}".format(scores.std()))

#==========================================
#4) Busca Hiperparamétrica: Utilizar GridSearch para otimizar os hiperparâmetros dos modelos.
#==========================================

# Definir grade de hiperparâmetros a testar
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

# Reutilizar o validador k-fold estratificado
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Instanciar o modelo
modelo_base = DecisionTreeClassifier(random_state=42)

# Configurar o GridSearch
grid_search = GridSearchCV(
    estimator=modelo_base,
    param_grid=param_grid,
    cv=kfold,
    scoring='accuracy',
    n_jobs=-1  # usa todos os núcleos disponíveis
)

# Executar a busca hiperparamétrica
grid_search.fit(X_pca, y)

# Mostrar os melhores parâmetros encontrados
print("\n Melhores hiperparâmetros encontrados:")
print(grid_search.best_params_)

# Mostrar a melhor acurácia obtida
print("Melhor acurácia média:", round(grid_search.best_score_, 4))

#==========================================
#5) Pruning de Árvores de Decisão: Realizar o pruning (poda) em árvores de decisão para prevenir o overfitting e melhorar a generalização do modelo.
#==========================================

# Treinar modelo base (sem poda) para pegar os alphas possíveis
modelo_pruning_base = DecisionTreeClassifier(random_state=42)
path = modelo_pruning_base.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]  # remove o maior alpha (zera tudo)

# Treinar uma árvore para cada alpha
modelos_podados = []
scores_teste = []

for alpha in ccp_alphas:
    modelo_podado = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    modelo_podado.fit(X_train, y_train)
    score = modelo_podado.score(X_test, y_test)
    modelos_podados.append(modelo_podado)
    scores_teste.append(score)

# Plotar acurácia vs alpha
plt.figure(figsize=(8, 5))
plt.plot(ccp_alphas, scores_teste, marker='o')
plt.xlabel("ccp_alpha")
plt.ylabel("Acurácia no teste")
plt.title("Poda de Árvore: Acurácia vs ccp_alpha")
plt.grid()
plt.show()

# Melhor modelo podado
melhor_indice = scores_teste.index(max(scores_teste))
melhor_alpha = ccp_alphas[melhor_indice]
melhor_modelo_podado = modelos_podados[melhor_indice]

print(f"\n Melhor ccp_alpha encontrado: {melhor_alpha}")
print("Acurácia no teste com poda:", round(scores_teste[melhor_indice], 4))

#==========================================
#6) Avaliação de Classificadores Binários: Utilizar figuras de mérito como Curva ROC, precisão, recall, f1-score, sensibilidade e especificidade para avaliar os modelos.
#==========================================

# Prever com o melhor modelo podado
y_pred_podado = melhor_modelo_podado.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_podado)
tn, fp, fn, tp = cm.ravel()

# Cálculo das métricas
precisao = precision_score(y_test, y_pred_podado, pos_label='M')
recall = recall_score(y_test, y_pred_podado, pos_label='M')  # sensibilidade
f1 = f1_score(y_test, y_pred_podado, pos_label='M')
especificidade = tn / (tn + fp)
acuracia = accuracy_score(y_test, y_pred_podado)

print("\n Avaliação do Melhor Modelo Podado:")
print("Acurácia:", round(acuracia, 4))
print("Precisão:", round(precisao, 4))
print("Recall (Sensibilidade):", round(recall, 4))
print("F1-score:", round(f1, 4))
print("Especificidade:", round(especificidade, 4))

# Curva ROC
# Converter y_test e y_pred_proba para binário (label M = positivo)
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)
y_score = melhor_modelo_podado.predict_proba(X_test)[:, lb.classes_.tolist().index('M')]

fpr, tpr, thresholds = roc_curve(y_test_bin, y_score)
roc_auc = auc(fpr, tpr)

# Plot da curva ROC
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"Curva ROC (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Aleatório")
plt.xlabel("Taxa de Falsos Positivos (1 - Especificidade)")
plt.ylabel("Taxa de Verdadeiros Positivos (Sensibilidade)")
plt.title("Curva ROC - Modelo Podado")
plt.legend(loc="lower right")
plt.grid()
plt.show()