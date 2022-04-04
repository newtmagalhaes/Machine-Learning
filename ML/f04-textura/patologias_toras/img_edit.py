import numpy as np
from scipy.stats import mode
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.transform import hough_line, hough_line_peaks, rotate

LIMITE_DE_NORMALIZACAO = 8


def best_rgb(img:np.ndarray) -> 'tuple[int, int]':
  """
  ## Parâmetros
  - img: imagem RGB do tipo ndarray com o shape (height, width, 3).

  ## Retorna
  Uma dupla com qualquer combinação de 0, 1 ou 2, índices associados
  respectivamente com R, G e B, indicando para cada eixo,
  respectivamente VERTICAL e HORIZONTAL, qual a cor com maior
  variação.
  """
  VERTICAL, HORIZONTAL = 0, 1
  cores = []
  for eixo in [VERTICAL, HORIZONTAL]:
    soma_R, soma_G, soma_B = img.sum(axis=eixo).transpose()
    variancias = np.array([arr.var() for arr in [soma_R, soma_G, soma_B]])
    cores.append(variancias.argmax())

  return tuple(cores)


def rgb_to_color(img:np.ndarray, color:int) -> np.ndarray:
  """RGB para mono
  ================
  Cria uma imagem monocromática partir de outra RGB com apenas uma das cores.

  ## Parâmetros
  - img: ndarray de shape (height, width, 3);
  - color: {0, 1, 2}, indicando respectivamente qual cor: R, G ou B
  se deseja criar a nova imagem.

  ## Retorna
  Uma nova imagem 2D onde cada píxel na coordenada x, y contém o
  respectivo valor com cor indicada
  (`img[x, y, color] == new_img[x, y]`).
  """
  height, width, _ = img.shape
  new_img = np.zeros(shape=(height, width))
  for i in range(height):
    for j in range(width):
      new_img[i, j] = img[i, j, color]
  
  return new_img


def _find_tilt_angle(image_edges:np.ndarray) -> float:
  """Encontrar ângulo de inclinação
  =================================
  Recebe uma imagem 2D binarizada (após ser tratada pelo `canny`).

  ## Retorna
  Um ângulo em graus de uma linha identficada pela transformada de
  Hough.
  """
  h, theta, d = hough_line(image_edges)
  accum, angles, dists = hough_line_peaks(h, theta, d)
  angle = np.rad2deg(mode(angles)[0][0])

  return angle + 90 if angle < 0 else angle - 90


def _find_best_angle(image_mono:np.ndarray) -> 'tuple[float, int]':
  """Encontrar melhor ângulo
  ==========================
  Versão otmizada de `_find_tilt_angle`.

  Retorna
  -------
  Ângulo encontrado em graus
  """
  # constantes definidas empiricamente
  CANNY_SIGMA = 2.0
  SIGMA_INCREMENT = 0.5
  SIGMA_RANGE = np.arange(CANNY_SIGMA, 5.5, SIGMA_INCREMENT) # [2, 2.5, ..., 5]

  moda_anterior = 0
  best_result = {'moda':0, 'total':0, 'angulo':0}
  
  # método de decisão para o melhor resultado
  _decide = lambda r: (-r['moda'], -r['total'], abs(r['angulo']))
  
  # encontrando bordas
  img_otsu = image_mono >= threshold_otsu(image_mono)

  for canny_sigma in SIGMA_RANGE:
    edges = canny(img_otsu, sigma=canny_sigma)
    h, theta, d = hough_line(edges)
    _, angles, dists = hough_line_peaks(h, theta, d)

    # calculando estatísticas
    moda = mode(angles, axis=None)

    actual_result = {'moda':moda.count[0], 'total':len(angles), 'angulo':moda.mode[0]}

    best_result = sorted([best_result, actual_result],
                          key=_decide)[0]
  
    # acima de 2 empiricamente se mostrou um bom resultado
    # se a moda atual for pior que a anterior, entende-se que não melhorará
    if actual_result['moda'] > 2 or actual_result['moda'] < moda_anterior:
      break
    moda_anterior = actual_result['moda']

  angle = np.rad2deg(best_result['angulo'])
  return angle + 90 if angle < 0 else angle - 90


def crop_empty_edges(img:np.ndarray) -> np.ndarray:
  """
  Dado uma imagem 2D que após ser rotacionada apresenta "triângulos
  pretos" em suas bordas, busca cortar essas partes da imagem.
  
  ## Parâmetros
  - img: uma matriz 2D representando a imagem;

  ## Retorna
  Uma nova imagem 2D, um recorte da original.
  """
  CANTOS = ['top_left', 'top_right', 'bot_left', 'bot_right']
  BORDAS_DICT = {s:0 if i < 2 else len(img) - 1 for i, s in enumerate(CANTOS)}

  for i, edge in enumerate(BORDAS_DICT):
    e = - (i % 2)
    while img[BORDAS_DICT[edge], e] == 0:
      BORDAS_DICT[edge] += 1 if i < 2 else -1

  max_top = max(BORDAS_DICT['top_left'], BORDAS_DICT['top_right'])
  min_bot = min(BORDAS_DICT['bot_left'], BORDAS_DICT['bot_right'])
  
  return img[max_top:min_bot+1].copy()


def fill_empty_edges(img:np.ndarray, metodo:int=0) -> np.ndarray:
  """Preencher cantos vazios
  ==========================
  Recebe uma imagem (rotacionada) que possua vazios nos cantos devido
  à rotação aplicada para serem preenchidos.

  ## Parâmetros
  - `img`: imagem, representada por uma matriz 2D onde cada elemento é:
    * um número: quando a imagem está em escala de cinza, por exemplo;
    * uma tripla: onde cada número da tripla representa a intensidade
    de cada cor (RGB, respectivavmente);
  
  - `metodo`: inteiro definindo qual método será aplicado para
  preencher os vazios presentes nas bordas:
    * 0: preencher com média da intensidade dos píxels;
    * 1: preencher com cópia da imagem ao lado;
    * outro: opção inválida, nada será feito.
  """
  # (h) altura e (w) comprimento da imagem
  h, w = np.array(img.shape[0:2]) - 1
  new_img:np.ndarray = img.copy()

  # Sistema de coordenadas da imagem tem origem (0, 0) no canto
  # superior esquerdo da imagem e cresce para a direita e para
  # baixo.
  # 'canto': row_start, row_direcao, col_start, col_direcao.
  CANTOS = {'top_left' :(0,  1, 0,  1),
            'top_right':(0,  1, w, -1),
            'bot_left' :(h, -1, 0,  1),
            'bot_right':(h, -1, w, -1)}
  
  if metodo == 0:
    # preencher com média da intensidade dos píxels da imagem
    _met = lambda linha=0, coluna=0, col_inicio=0, passo=1: new_img.mean()
  elif metodo == 1:
    # preenche com cópia da imagem ao lado
    _met = lambda linha=0, coluna=0, col_inicio=0, passo=1: new_img[linha, coluna:2*coluna-col_inicio:passo]
  else:
    # opção inválida, nada será feito
    print('método errado!!')
    return new_img

  for canto in CANTOS:
    row_start, row_direcao, col_start, col_direcao = CANTOS[canto]
    row, col = row_start, col_start # contadores
    
    # Enquanto contadores estão no intervalo \
    # AND o píxel atual for preto (grayscale:0 e RGB:(0,0,0))
    while (0 <= row <= h) and new_img[row, col].all() == 0:
      while (0 <= col <= w) and new_img[row, col].all() == 0:
        col += col_direcao

      # Preenche uma linha horizontal do canto por vez
      new_img[row, col_start:col:col_direcao] = _met(row, col, col_start, col_direcao)

      col = col_start
      row += row_direcao
  
  return new_img


def crop_horizontal(img:np.ndarray,
                    soma_horizontal:np.ndarray,
                    metodo:int=0,
                    norm:int=LIMITE_DE_NORMALIZACAO,
                    return_thresholds:bool=False) -> 'np.ndarray | tuple[int, int]':
  """Crop horizontal
  ==================
  ## Parâmetros
  - `img`: imagem, representada como uma matriz 2D
  - `soma_horizontal`: um array (1D) onde o elemento `i` corresponde ao
  somatório da linha `i` (equivale a `img.sum(axis=1)`)
  - `metodo`: inteiro definindo qual método será aplicado para definir
  o ajuste (+ ou -, na direção onde os dados se concentram):
    * 0: 0.5
    * 1: `média+variância - base` (base = início do intervalo do pico)
    * outro: sem ajuste
  - `norm`: Constante para normalização da `soma_horizontal`
  - `return_thresholds`: opção de retorno (False por padrão):
    * True: tupla com os pontos de corte da imagem;
    * False: cópia da imagem recortada.
  """
  soma_normalizada = soma_horizontal * (norm - 1)/soma_horizontal.max()
  n = len(soma_horizontal) - 1
  hist, bins = np.histogram(soma_normalizada, range(LIMITE_DE_NORMALIZACAO), density=False)
  
  # intervalo = início do intervalo do pico do histograma SE (pos < n/2)
  # intervalo = fim do intervalo do pico do histograma SE (pos >= n/2)
  # intervalo = bins[pos + int(pos < n/2)]
  pos = hist.argmax()
  aux = soma_normalizada[soma_normalizada>=bins[pos]]
  bar:np.ndarray = aux[aux<bins[pos+1]]
  
  if metodo == 0:
    ajuste = 0.5
  elif metodo == 1:
    ajuste = bar.mean() + bar.var() - bins[pos]
  else:
    ajuste = 0

  # Verificando onde se concentram os dados
  esquerda = hist[:pos].sum()
  direita = hist[pos+1:].sum()
  if esquerda >= direita:
    ajuste *= -1
    intervalo = bins[pos]
    stop_cond = lambda x, y_bar: x < y_bar
  else:
    intervalo = bins[pos+1]
    stop_cond = lambda x, y_bar: x > y_bar
  
  limiar = intervalo + ajuste

  # Percorrendo imagem para traçar corte
  # de cima para baixo
  pos_corte = [0, n]
  for j, v in enumerate(soma_normalizada):
    if stop_cond(v, limiar):
      pos_corte[0] = j
      break
  # de baixo para cima
  for j, v in enumerate(soma_normalizada[::-1]):
    if stop_cond(v, limiar):
      pos_corte[1] = n - j
      break
  
  if return_thresholds:
    return pos_corte
  else:
    start, stop = pos_corte
    return img[start:stop].copy()


def auto_rotate(img:np.ndarray) -> 'tuple[np.ndarray, float]':
  """
  Dado uma imagem 2D, binariza com limiar de OTSU, passa pelo filtro
  de canny, rotaciona com o ângulo calculado pela transformada de
  Hough e corta os espaços vazios gerados pela rotação
  
  ## Parâmetros
  - img: imagem 2D.

  ## Retorna
  Uma imagem e o ângulo de rotação (em graus), no qual a imagem é:
  - uma nova, rotacionada e recortada se o ângulo identificado for
  diferente de 0;
  - a mesma imagem se o ângulo identificado for 0.
  """
  # binarizando com otsu
  img_ostu = img >= threshold_otsu(img)

  # encontrando bordas
  edges = canny(img_ostu)

  # Rotacionando imagem se for preciso
  angle = _find_tilt_angle(edges)
  if angle != 0:
    new_img = rotate(img, angle)
    # crop_img = crop_empty_edges(new_img)
    return new_img, angle
  else:
    return img, 0


if __name__ == '__main__':
  a = [1, 2, 3]
  print(f'a: {a}\t-\ttipo: {type(a)}')
  b = tuple(a)
  print(f'b: {b}\t-\ttipo: {type(b)}')
