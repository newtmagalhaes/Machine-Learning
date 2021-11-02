import numpy as np
from typing import Tuple
from scipy.stats import mode
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.transform import hough_line, hough_line_peaks, rotate


def best_rgb(img:np.ndarray) -> Tuple[int, int]:
  soma_vertical_R, soma_vertical_G, soma_vertical_B = img.sum(axis=0).transpose()
  soma_horizontal_R, soma_horizontal_G, soma_horizontal_B = img.sum(axis=1).transpose()
  
  v = [arr.max() - arr.min() for arr in [soma_vertical_R, soma_vertical_G, soma_vertical_B]]
  max_diff = 0
  for i, diff in enumerate(v):
    if diff > max_diff:
      max_diff = diff
      best_v = i

  h = [arr.max() - arr.min() for arr in [soma_horizontal_R, soma_horizontal_G, soma_horizontal_B]]
  max_diff = 0
  for i, diff in enumerate(h):
    if diff > max_diff:
      max_diff = diff
      best_h = i

  return best_v, best_h


def find_tilt_angle(image_edges:np.ndarray) -> float:
  h, theta, d = hough_line(image_edges)
  accum, angles, dists = hough_line_peaks(h, theta, d)
  angle = np.rad2deg(mode(angles)[0][0])

  return angle + 90 if angle < 0 else angle - 90


def crop_empty_edges(img:np.ndarray) -> np.ndarray:
  '''
  ---
  - img: np.ndarray 2D image
  '''
  top_left, top_right = 0, 0
  while img[top_left, 0] == 0:
    top_left += 1
  while img[top_right, -1] == 0:
    top_right += 1
  
  bot_left, bot_right = len(img) - 1, len(img) - 1
  while img[bot_left, 0] == 0:
    bot_left -= 1
  while img[bot_right, 0] == 0:
    bot_right -= 1

  max_top = max(top_left, top_right)
  min_bot = min(bot_left, bot_right)
  new_height = min_bot - max_top
  width = len(img[0])
  new_img = np.zeros((new_height, width))

  for i in range(new_height):
    for j in range(width):
      new_img[i, j] = img[i + max_top, j]
  
  return new_img


def rgb_to_color(img:np.ndarray, color:int) -> np.ndarray:
  height, width, _ = img.shape
  new_img = np.zeros(shape=(height, width))

  for i in range(height):
    for j in range(width):
      new_img[i, j] = img[i, j, color]
  
  return new_img

def auto_rotate_and_crop(img:np.ndarray) -> Tuple[np.ndarray, float]:
  '''
  ---
  - img: np.ndarray 2D image
  '''
  # binarizando com otsu
  img_ostu = img >= threshold_otsu(img)

  # encontrando bordas
  edges = canny(img_ostu)

  # Rotacionando imagem
  angle = find_tilt_angle(edges)
  new_img = rotate(img, angle)

  crop_img = crop_empty_edges(new_img)
  return crop_img, angle
