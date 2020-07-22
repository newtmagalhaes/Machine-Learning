# %% [markdown]
# # CHB dataset
#
# Script para gerar dataset `chb01.csv`
#
# > `neg` e `pos` em variáveis se referem às classes (`target`):
#
# - 0: negativo
# - 1: positivo
#

# %%
# Importando bibliotecas
import numpy as np
import pandas as pd

from zipfile import ZipFile
from statsmodels.robust.scale import mad as medianAD

# %%
features = ['std', 'mean', 'skew', 'kurt', 'meanAD', 'medianAD', 'energy']

# Colunas do Dataset
labels = [
    f'{feat}-{column}' for column in range(18) for feat in features
] + ['target']


def extract_statistical_features(matrix: np.ndarray, target: np.float64) -> pd.DataFrame:
    '''
    É esperado como entrada uma matrix de dimensões 18 x 512.

    Durante a execução, um dataframe statsDF de dimensões 7 x 18

    Ao fim um DataFrame com 127 colunas e 1 linha
    126 (7 * 18 do statsDF) + 1 (do target)
    '''

    def energy(mat): return (mat ** 2).sum(axis=1)

    matrixDF, statsDF = pd.DataFrame(data=matrix).transpose(), pd.DataFrame()

    statsDF['std'] = matrixDF.std()
    statsDF['mean'] = matrixDF.mean()
    statsDF['skew'] = matrixDF.skew()
    statsDF['kurt'] = matrixDF.kurt()
    statsDF['meanAD'] = matrixDF.mad()
    statsDF['medianAD'] = medianAD(matrixDF)
    statsDF['energy'] = energy(matrix)

    # Redimensionando matriz para uma linha e concatenando target à lista
    column = list(statsDF.values.reshape(126)) + [target]

    return pd.DataFrame(data=column, index=labels).transpose()


# %%
# Carregando Matrizes de arquivo zip
with ZipFile('chb01.zip', 'r') as data:
    # Cria uma lista com os nomes dos arquivos dentro do zip e os ordena
    file_list = data.namelist()
    file_list.sort()

    pos_list = [pos for pos in file_list if ('chb01/positive/' in pos)]
    neg_list = [file_list[i] for i in range(len(pos_list))]

    print(f'pos_list: {len(pos_list)}\tneg_list: {len(neg_list)}')

    pos_space, neg_space = [], []

    # Cada arquivo é uma matriz que será salva nas listas {pos, neg}_space
    for pos_file, neg_file in zip(pos_list, neg_list):
        with data.open(name=pos_file, mode='r') as pos, data.open(name=neg_file, mode='r') as neg:
            pos_space.append(np.load(pos))
            neg_space.append(np.load(neg))

    # Convertendo listas para arrays
    pos_space = np.array(pos_space, dtype=np.float64)
    neg_space = np.array(neg_space, dtype=np.float64)

    print(
        f'pos_space.shape: {pos_space.shape}\tneg_space.shape: {neg_space.shape}')
    print(
        f'pos_space.dtype: {pos_space.dtype}\tneg_space.dtype: {neg_space.dtype}')

# %%
# Gerando dataframe
dataset = pd.DataFrame(columns=labels)

for neg_matrix in neg_space:
    dataset = dataset.append(
        extract_statistical_features(matrix=neg_matrix, target=0),
        ignore_index=True
    )

for pos_matrix in pos_space:
    dataset = dataset.append(
        extract_statistical_features(matrix=pos_matrix, target=1),
        ignore_index=True
    )
# %%
# Salvando dataset em arquivo csv
dataset.to_csv(path_or_buf='chb01.csv', index=False)
