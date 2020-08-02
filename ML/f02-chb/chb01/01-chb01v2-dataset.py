# %% [markdown]
# # CHB dataset 2
#
# Script para gerar dataset `chb01dwt.csv`
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

from pywt import wavedec
from zipfile import ZipFile
from statsmodels.robust.scale import mad as medianAD


# %%
# Colunas do Dataset
decomposition_level = 5

channels = [column for column in range(18)]

dwt_coefs = [f'A{decomposition_level}'] + \
    [f'D{level}' for level in range(1, decomposition_level + 1)]

stats = ['std', 'mean', 'skew', 'kurt', 'meanAD', 'medianAD', 'energy']

dataset_labels = [
    f'{feat}-{coef}-{column}' for column in channels for coef in dwt_coefs for feat in stats
] + ['target']


def energy(arr: np.ndarray) -> np.float64: return (arr ** 2).sum()


def extract_statistical_features(matrix: np.ndarray, target: np.float64) -> pd.DataFrame:
    '''
    É esperado como entrada uma matrix de dimensões 18 x 512.

    Durante a execução, para cada uma das 18 colunas,
    é gerado uma lista de 6 arrays de tamanhos variados
    em função da wavedec.
    De cada um dos 6 arrays é extraido 7 features.

    um dataframe statsDF de dimensões 1 x 757 é retornado
    com as features calculadas.
    '''
    statsDF = pd.DataFrame(columns=dataset_labels)

    for i, column in enumerate(matrix):
        wavelet_coefs = wavedec(
            data=column,
            wavelet='db2',
            level=decomposition_level
        )

        for coef, coef_label in zip(wavelet_coefs, dwt_coefs):
            coefDF = pd.DataFrame(data=coef, dtype=np.float64)

            statsDF[f'std-{coef_label}-{i}'] = coefDF.std()
            statsDF[f'mean-{coef_label}-{i}'] = coefDF.mean()
            statsDF[f'skew-{coef_label}-{i}'] = coefDF.skew()
            statsDF[f'kurt-{coef_label}-{i}'] = coefDF.kurt()
            statsDF[f'meanAD-{coef_label}-{i}'] = coefDF.mad()
            statsDF[f'medianAD-{coef_label}-{i}'] = medianAD(coefDF)
            statsDF[f'energy-{coef_label}-{i}'] = energy(coef)

    statsDF['target'] = target
    return statsDF


# %%
# Carregando matrizes de arquivo zip
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
# Extraindo atributos e gerando dataset
dataset = pd.DataFrame(columns=dataset_labels)

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
dataset.to_csv(path_or_buf='chb01dwt.csv', index=False)
