import numpy as np
import pandas as pd
from pywt import wavedec
from zipfile import ZipFile
from statsmodels.robust.scale import mad as medianAD


def get_class_and_frequence(path: str) -> (int, int):
    '''
	`path` é uma str no modelo: 'pasta/subpasta/arquivo'.
	
	O retorno é uma tupla contendo `(classe, frequência)`,
	onde os valores estão presentes nos nomes da subpasta
	e arquivo, respectivamente.
    '''
    _, class_str, freq_str = path.split('/')

	# A classe é o ultimo caractere da string
    class_int = int(class_str[-1])

	# O nome do arquivo separa 4 valores pelo char 'c' (V0cV1cV2cV3.csv)
    # No qual a frequência é o terceiro valor, V2
    freq_int = int(freq_str.split('c')[2])
    
    return (class_int, freq_int)

def energy(vec:np.ndarray) -> np.float64:
	return np.square(vec).sum()

def create_fs20(vec:np.ndarray, file_path:str) -> pd.DataFrame:
    '''
    Dado um sinal (`vec`) e o nome do arquivo de origem (`file_path`),
    retorna um dataframe de 1 linha com os atributos do "Feature Set 20" extraidos.
    
    Feature Set 20:
    ---
    + MeanAD D3, MeanAD D4, MeanAD A5;
    + MedianAD D3, MedianAD D4, MedianAD D5, MedianAD A5;
    + Energia D3, Energia D4, Energia D5, Energia A5;
    + Kurt D5, Kurt A5;
    + Skew D4
    + Frequency;
    '''
    result_df = pd.DataFrame()

    # tupla de coeficientes: (A5, D5, D4, ..., D1)
    dwt_coefs = wavedec(data=vec, wavelet='db2', level=5)

    # meanAD A5, D4, D3
    for index, coef in zip([0, 2, 3], ['A5', 'D4', 'D3']):
        result_df[f'MeanAD-{coef}'] = pd.DataFrame(dwt_coefs[index]).mad()

    # medianAD A5, D5, D4, D3 e Energia A5, D5, D4, D3
    for index, coef in zip([0, 1, 2, 3], ['A5', 'D5', 'D4', 'D3']):
        result_df[f'MedianAD-{coef}'] = medianAD(dwt_coefs[index])
        result_df[f'Energy-{coef}'] = energy(dwt_coefs[index])
    
    # Kurtosis A5
    result_df['Kurt-A5'] = pd.DataFrame(dwt_coefs[0]).kurt()

    # Kurtosis D5
    result_df['Kurt-D5'] = pd.DataFrame(dwt_coefs[1]).kurt()

    # Skewness D4
    result_df['Skew-D4'] = pd.DataFrame(dwt_coefs[2]).skew()

    # target e Frequence
    target, frequence = get_class_and_frequence(file_path)
    result_df['frequence'] = frequence
    result_df['target'] = target
    
    return result_df.copy()
