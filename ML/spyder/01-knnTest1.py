# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # KNN Test
# 
# Testando o algoritmo KNN no [IRIS dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set "wiki of iris flower data set") utilizando as bibliotecas `scikit-learn`, `pandas` e `matplotlib`.
# 
# Seguindo o tutorial do [video](https://www.youtube.com/watch?v=DeAuVrhKw58).
# %% [markdown]
# ## Importando Bibliotecas

# %%
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# %% [markdown]
# ## Carregando IRIS _DataSet_
# 
# Usando `scikit-learn`

# %%
iris_data, iris_targets = load_iris(return_X_y=True)
iris_dset = load_iris(return_X_y=False)

print("Target names:\n", list(iris_dset.target_names))

print("\nFeature names:\n", list(iris_dset.feature_names))

# %% [markdown]
# ## Separando dados para treino e teste

# %%
data_train, data_test, target_train, target_test = train_test_split(iris_data, iris_targets, test_size=0.30, random_state=13)

# %% [markdown]
# ## Instanciando classificador

# %%
k_neighbors = 13

classifyer = KNeighborsClassifier(n_neighbors=k_neighbors)

# %% [markdown]
# ## Treinando modelo classificador

# %%
classifyer.fit(data_train, target_train)

# %% [markdown]
# ## Realizando teste de classificação

# %%
prediction = classifyer.predict(data_test)

# %% [markdown]
# ## Resultados

# %%
print(classification_report(target_test, prediction, target_names=iris_dset.target_names))


# %%
confusion_matrix(target_test, prediction)

