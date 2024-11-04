import numpy as np
from skimage import color, io
from skimage.transform import resize
import cv2

import matplotlib.pyplot as plt
import os
import random
import time

from typing import List, Tuple,Union

from heapq import nsmallest
class Cell:
    def __init__(self, i: int, j: int, cost: np.ndarray, before):
        self.i, self.j = i, j
        self.cost = cost 
        self.before = before 
    @property
    def point(self) -> Tuple[int, int]:
        return (self.i, self.j)
    
class Grid:
    def __init__(self, n: int, m: int):
        self.n, self.m = n, m
        self.d = []
    def add(self, c: Cell) -> None:
        self.d.append(c)
    def get(self, i: int, j: int) -> List[Cell]:
        return [c for c in self.d if (c.i == i) and (c.j == j)][0]
    def get_col(self, j: int) -> List[Cell]:
        return self.d[self.n*j:self.n*2*j]

def _ssd_patch(T: np.ndarray, M: np.ndarray, I: np.ndarray) -> np.ndarray:
    """
    Performes template matching with the overlapping region, computing the cost of sampling each patch, based on the sum of squared differences (SSD).

    :param T: Patch in the current output image that is to be filled in
    :param M: Mask of the overlapping region
    :param I: Sample image
    :return: SSD of the patch
    """
    T = T.astype(np.float64)
    M = M.astype(np.float64)
    I = I.astype(np.float64)
    
    def _ssd(ch: int):
        return ((M[:,:,ch]*T[:,:,ch])**2).sum() - 2 * cv2.filter2D(I[:,:,ch], ddepth=-1, kernel = M[:,:,ch]*T[:,:,ch]) + cv2.filter2D(I[:,:,ch] ** 2, ddepth=-1, kernel=M[:,:,ch])

    ssd_b = _ssd(0)
    ssd_g = _ssd(1)
    ssd_r = _ssd(2)

    return ssd_b + ssd_g + ssd_r

def _choose_sample(cost: np.ndarray, tol: int) -> np.ndarray:
    """
    Selects a randomly sampled patch with low cost.

    :param cost: Cost matrix
    :param tol: Tolerance
    :return: Patch with min cost
    """
    idx = np.argpartition(cost.ravel(), tol-1)[:tol]
    lowest_cost = np.column_stack(np.unravel_index(idx, cost.shape))
    return random.choice(lowest_cost)
def customized_cut(err_patch: np.ndarray) -> Union[np.ndarray, List]:
    h, w = err_patch.shape[:2]
    grid = Grid(h, w)

    for i in range(h):
        grid.add(Cell(i, 0, err_patch[i, 0], None))

    for j in range(1, w):
        for i in range(h):
            if i - 1 < 0:
                before = [grid.get(i, j-1), grid.get(i+1, j-1)] if i + 1 < h else [grid.get(i, j-1)]
            elif i + 1 >= h:
                before = [grid.get(i-1, j-1), grid.get(i, j-1)]
            else:
                before = [grid.get(i-1, j-1), grid.get(i, j-1), grid.get(i+1, j-1)]

            # Check if before is empty
            if not before:
                raise ValueError(f"No valid cells found for the previous column at j={j}, i={i}.")

            min_before = min(before, key=lambda x: x.cost)

            cell = Cell(i, j, err_patch[i, j] + min_before.cost, min_before)
            grid.add(cell)

    last_column = grid.get_col(w-1)
    
    if not last_column:
        raise ValueError("Last column is empty, cannot find minimum path.")

    cell = min(last_column, key=lambda x: x.cost)
    mask = np.zeros(err_patch.shape)
    best_path = []

    for i in range(w, 0, -1):
        best_path.append(cell.point)
        x, y = cell.point
        mask[:x, y] = 1
        cell = cell.before

    return mask, best_path

def _multiply_on_last_axis(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    for i in range(matrix_a.shape[2]):
        matrix_a[:,:,i] *= matrix_b
    return matrix_a

def texture_transfer(input_image_file, target_image_file):
    # sample_img = cv2.cvtColor(cv2.imread(input_image_file), cv2.COLOR_BGR2RGB)
    # sample_trg = cv2.cvtColor(cv2.imread(target_image_file), cv2.COLOR_BGR2RGB)
    input_image = io.imread(input_image_file)
    target_image = io.imread(target_image_file)
    target_w, target_h = target_image.shape[:2]
    output = np.zeros((target_w, target_h, 3))
  

    print(input_image)
    patch_size = 20
    overlap = 12
    tol = 50
    iter_num = 3
    reduction = 0.7
    
    for n in range(iter_num):
        synthesized_output = output.copy()
        output = np.zeros((target_w, target_h, 3))
        alpha = 0.1 + 0.8 * n / (iter_num - 1)
        offset = patch_size - overlap
        half = patch_size // 2
        for i in range(0, target_w-offset, offset):
            for j in range(0, target_h-offset, offset):
                print(i,"/",target_w-offset," ",j,"/",target_h-offset)
               
                template = output[i:i+patch_size, j:j+patch_size, :].copy()
                _target = target_image[i:i+patch_size, j:j+patch_size, :].copy()
                synthesized = synthesized_output[i:i+patch_size, j:j+patch_size, :].copy()
                
                mask = np.zeros((patch_size, patch_size, 3))

                if template.shape[:2] != (patch_size, patch_size):
                    continue
                if i == 0:
                    mask[:, :overlap, :] = 1  # upper row mask
                elif j == 0:
                    mask[:overlap, :, :] = 1  # left column mask
                else:
                    mask[:, :overlap, :] = 1
                    mask[:overlap, :, :] = 1
                
                ssd_overlap = _ssd_patch(template, mask, input_image)
                ssd_overlap = ssd_overlap[half:-half, half:-half]  
                
                ssd_target = _ssd_patch(_target, np.ones((patch_size, patch_size, 3)), input_image)
                ssd_target = ssd_target[half:-half, half:-half] 
               
                ssd_prev = 0
                if n > 0:  # n is number of iteration
                    ssd_prev = _ssd_patch(synthesized, np.ones((patch_size, patch_size, 3)), input_image)
                    ssd_prev = ssd_prev[half:-half, half:-half] 
                
                ssd = (ssd_overlap + ssd_prev) * alpha + ssd_target * (1 - alpha)
                x, y = _choose_sample(ssd, tol)

                patch = input_image[x:x+patch_size, y:y+patch_size, :].copy()
                mask1 = np.zeros((patch_size, patch_size))
                diff1 = (template[:overlap, :patch_size, :] - patch[:overlap, :patch_size, :]) ** 2
                diff1 = np.sum(diff1, axis=2)
                mask_patch1, _ = customized_cut(diff1)
                mask1[:overlap, :patch_size] = mask_patch1

                mask2 = np.zeros((patch_size, patch_size))
                diff2 = (template[:patch_size, :overlap, :] - patch[:patch_size, :overlap, :]) ** 2
                diff2 = np.sum(diff2, axis=2)
                mask_patch2, _ = customized_cut(diff2.T)
                mask2[:patch_size, :overlap] = mask_patch2.T
                if j == 0:
                    mask_cut = mask1.copy()
                elif i == 0:
                    mask_cut = mask2.copy()
                else:
                    mask_cut = np.logical_or(mask1.copy(), mask2.copy())
                mask_cut = mask_cut.astype(np.uint8)
            
                template = _multiply_on_last_axis(template.copy(), mask_cut.copy())

                mask_cut ^= 1
                patch = _multiply_on_last_axis(patch.copy(), mask_cut.copy())

                output[i:i+patch_size, j:j+patch_size, :] = patch + template
        patch_size = int(patch_size * reduction)
        overlap = int(overlap * reduction)

                
    return output.astype(np.uint8)



   