import numpy as np
import pandas as pd

from zipfile import ZipFile
from statsmodels.robust.scale import mad as medianAD

features = ['std', 'mean', 'skew', 'kurt', 'meanAD', 'medianAD', 'energy']

# Colunas do Dataset com DWT
decomposition_level = 5

dwt_coefs = [f'A{decomposition_level}'] + [f'D{level}' for level in range(1, decomposition_level + 1)]

labels_com_dwt = [
    f'{feat}-{coef}-{column}' for column in range(18) for coef in dwt_coefs for feat in features
] + ['target']

# Colunas do Dataset sem DWT
labels_sem_dwt = [
    f'{feat}-{column}' for column in range(18) for feat in features
] + ['target']


def energy(mat: np.ndarray) -> np.float64: return (mat ** 2).sum(axis=1)


def extract_features_sem_dwt(matrix: np.ndarray, target: np.float64) -> pd.DataFrame:
    '''
    É esperado como entrada uma matrix de dimensões 18 x 512.

    Durante a execução, um dataframe statsDF de dimensões 7 x 18

    Ao fim um DataFrame com 127 colunas e 1 linha
    126 (7 * 18 do statsDF) + 1 (do target)
    '''

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

    return pd.DataFrame(data=column, index=labels_sem_dwt).transpose()


def extract_features_com_dwt(matrix: np.ndarray, target: np.float64) -> pd.DataFrame:
    '''
    É esperado como entrada uma matrix de dimensões 18 x 512.

    Durante a execução, para cada uma das 18 colunas,
    é gerado uma lista de 6 arrays de tamanhos variados
    em função da wavedec.
    De cada um dos 6 arrays é extraido 7 features.

    um dataframe statsDF de dimensões 1 x 757 é retornado
    com as features calculadas.
    '''
    statsDF = pd.DataFrame(columns=labels_com_dwt)

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
