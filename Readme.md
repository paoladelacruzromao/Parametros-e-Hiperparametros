# Parametros-e-Hiperparametros

Com  método Ensemble eu vou precissar tratar un número maior de modelos  para isso precisamos de estrategias para tratar os parametros do estimador base e do modelo ensemble para encontrar a melhor combinação precisamos aplicar a estrategia da otimização dos hiperparametros.

Todo modelo de Machine Learning possui parâmetros que permitem a customização do modelo. Esses parâmetros também são chamados de hiperparâmetros
Em programação os algoritmos de Machine Learning são representados por funções e cada função possui os parâmetros de customização, exatamente o que chamamos de hiperparâmetros.

É comum ainda que as pessoas se refiram aos coeficientes do modelo (encontrados ao final do treinamento) como parâmetros.

Parte do nosso trabalho como Cientistas de Dados é encontrar a melhor combinação de hiperparâmetros para cada modelo.

Em Métodos Ensemble esse trabalho é ainda mais complexo, pois temos os hiperparâmetros do estimador base e os hiperparâmetros do modelo ensemble, conforme este exemplo abaixo:

Estimador base:
estim_base = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=None, n_neighbors=5, p=2, weights='uniform')

Modelo Ensemble:
BaggingClassifier(base_estimator=estim_base, bootstrap=True, bootstrap_features=False, max_features=0.5, max_samples=0.5, n_estimators=10, n_jobs=None, oob_score=False, random_state=None, verbose=0, warm_start=False)

## Extremely Randomized Forest

Modelo padrão, com hiperparâmetros escolhidos manualmente.
```sh
# Imports
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Carrega o dataset
data = pd.read_excel('dados/credit.xls', skiprows = 1)

print(data)

# Variável target
target = 'default payment next month'
y = np.asarray(data[target])

# Variáveis preditoras > eliminamos o ID por não ser relevante, e avariavek target doas variaveis de entrada
features = data.columns.drop(['ID', target])
X = np.asarray(data[features])

# Divisão de dados de treino e de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 99)

# Classificador
clf = ExtraTreesClassifier(n_estimators = 500, random_state = 99)

# Treinamento do Modelo
clf.fit(X_train, y_train)

# Score
scores = cross_val_score(clf, X_train, y_train, cv = 3, scoring = 'accuracy', n_jobs = -1)

# Imprimindo o resultado
print ("ExtraTreesClassifier -> Acurácia em Treino: Média = %0.3f Desvio Padrão = %0.3f" % (np.mean(scores), np.std(scores)))

# Fazendo previsões
y_pred = clf.predict(X_test)

# Confusion Matrix
confusionMatrix = confusion_matrix(y_test, y_pred)
print (confusionMatrix)

# Acurácia
print("Acurácia em Teste:", accuracy_score(y_test, y_pred))
```
------------------------------------------------------------------------------------------------------------------------------------------------------------

# Otimização dos Hiperparâmetros com Randomized Search

http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

O Randomized Search gera amostras dos parâmetros dos algoritmos a partir de uma distribuição randômica uniforme para um número fixo de interações. Um modelo é construído e testado para cada combinação de parâmetros.
```sh
# Import
from sklearn.model_selection import RandomizedSearchCV
# Definição dos parâmetros
param_dist = {"max_depth": [1, 3, 7, 8, 12, None],
              "max_features": [8, 9, 10, 11, 16, 22],
              "min_samples_split": [8, 10, 11, 14, 16, 19],
              "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7],
              "bootstrap": [True, False]}
 # Para o classificador criado com ExtraTrees, testamos diferentes combinações de parâmetros
rsearch = RandomizedSearchCV(clf, param_distributions = param_dist, n_iter = 25, return_train_score = True)  

# Aplicando o resultado ao conjunto de dados de treino e obtendo o score
rsearch.fit(X_train, y_train)

# Resultados 
rsearch.cv_results_

# Imprimindo o melhor estimador
bestclf = rsearch.best_estimator_
print (bestclf)

# Aplicando o melhor estimador para realizar as previsões
y_pred = bestclf.predict(X_test)

# Confusion Matrix
confusionMatrix = confusion_matrix(y_test, y_pred)
print(confusionMatrix)

# Acurácia
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Obtendo o grid com todas as combinações de parâmetros
rsearch.cv_results_
```
------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Grid Search x Randomized Search para Estimação dos Hiperparâmetros
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

O Grid Search realiza metodicamente combinações entre todos os parâmetros do algoritmo, criando um grid.
```sh
import numpy as np
from time import time
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits

# Obtém o dataset
digits = load_digits()
X, y = digits.data, digits.target

# Construindo o classificador
clf = RandomForestClassifier(n_estimators = 20)
```
### Executando Randomized Search

Usando  ***sp_randint*** para escolher os parametros de forma random
```sh

# Randomized Search

# Valores dos parâmetros que serão testados
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# Executando o Randomized Search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, 
                                   param_distributions = param_dist, 
                                   n_iter = n_iter_search,
                                   return_train_score=True)

start = time()
random_search.fit(X, y)
print("RandomizedSearchCV executou em %.2f segundos para %d candidatos a parâmetros do modelo." 
      % ((time() - start), n_iter_search))

# Imprime as combinações dos parâmetros e suas respectivas médias de acurácia
random_search.cv_results_
```
### Executando Grid Search

```sh
# Grid Search

# Usando um grid completo de todos os parâmetros
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# Executando o Grid Search
grid_search = GridSearchCV(clf, param_grid = param_grid, return_train_score = True)
start = time()
grid_search.fit(X, y)

print("GridSearchCV executou em %.2f segundos para todas as combinações de candidatos a parâmetros do modelo."
      % (time() - start))
grid_search.cv_results_

```
### FIM
