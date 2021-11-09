import numpy as np
from scipy.stats import mode
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.transform import hough_line, hough_line_peaks, rotate


def best_rgb(img:np.ndarray) -> 'tuple[int, int]':
  '''
  ## Parâmetros
  - img: imagem RGB do tipo ndarray com o shape (height, width, 3).

  ## Retorna
  Uma dupla com qualquer combinação de 0, 1 ou 2, índices associados
  respectivamente com R, G e B, indicando para cada eixo,
  respectivamente VERTICAL e HORIZONTAL, qual a cor com maior
  variação.
  '''
  VERTICAL, HORIZONTAL = 0, 1
  cores = []
  for eixo in [VERTICAL, HORIZONTAL]:
    soma_R, soma_G, soma_B = img.sum(axis=eixo).transpose()
    diferencas = [arr.max() - arr.min() for arr in [soma_R, soma_G, soma_B]]
    
    max_diff = 0
    for i, diff in enumerate(diferencas):
      if diff > max_diff:
        max_diff = diff
        melhor_cor = i

    cores.append(melhor_cor)

  return tuple(cores)


def find_tilt_angle(image_edges:np.ndarray) -> float:
  '''
  Recebe uma imagem 2D binarizada.

  ## Retorna
  Um ângulo em graus de uma linha identficada pela transformada de
  Hough.
  '''
  h, theta, d = hough_line(image_edges)
  accum, angles, dists = hough_line_peaks(h, theta, d)
  angle = np.rad2deg(mode(angles)[0][0])

  return angle + 90 if angle < 0 else angle - 90


def crop_empty_edges(img:np.ndarray) -> np.ndarray:
  '''
  Dado uma imagem 2D que após ser rotacionada apresenta "triângulos
  pretos" em suas bordas, busca cortar essas partes da imagem.
  
  ## Parâmetros
  - img: uma matriz 2D representando a imagem;

  ## Retorna
  Uma nova imagem 2D, um recorte da original.
  '''
  CANTOS = ['top_left', 'top_right', 'bot_left', 'bot_right']
  BORDAS_DICT = {s:0 if i < 2 else len(img) - 1 for i, s in enumerate(CANTOS)}

  for i, edge in enumerate(BORDAS_DICT):
    e = - (i % 2)
    while img[BORDAS_DICT[edge], e] == 0:
      BORDAS_DICT[edge] += 1 if i < 2 else -1

  max_top = max(BORDAS_DICT['top_left'], BORDAS_DICT['top_right'])
  min_bot = min(BORDAS_DICT['bot_left'], BORDAS_DICT['bot_right'])
  
  return img[max_top:min_bot+1].copy()


def rgb_to_color(img:np.ndarray, color:int) -> np.ndarray:
  '''
  Cria a partir de uma imagem RGB outra com apenas uma das cores.

  ## Parâmetros
  - img: ndarray de shape (height, width, 3);
  - color: {0, 1, 2}, indicando respectivamente qual cor: R, G ou B
  se deseja criar a nova imagem.

  ## Retorna
  Uma nova imagem 2D onde cada píxel na coordenada x, y contém o
  respectivo valor com cor indicada
  (`img[x, y, color] == new_img[x, y]`).
  '''
  height, width, _ = img.shape
  new_img = np.zeros(shape=(height, width))

  for i in range(height):
    for j in range(width):
      new_img[i, j] = img[i, j, color]
  
  return new_img

def auto_rotate_and_crop(img:np.ndarray) -> 'tuple[np.ndarray, float]':
  '''
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
  '''
  # binarizando com otsu
  img_ostu = img >= threshold_otsu(img)

  # encontrando bordas
  edges = canny(img_ostu)

  # Rotacionando imagem se for preciso
  angle = find_tilt_angle(edges)
  if angle != 0:
    new_img = rotate(img, angle)
    crop_img = crop_empty_edges(new_img)
    return crop_img, angle
  else:
    return img, 0


if __name__ == '__main__':
  a = [1, 2, 3]
  print(f'a: {a}\t-\ttipo: {type(a)}')
  b = tuple(a)
  print(f'b: {b}\t-\ttipo: {type(b)}')
