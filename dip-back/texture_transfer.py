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
from cell import Cell
from grid import Grid
def compute_patch_similarity(template: np.ndarray, overlap_mask: np.ndarray, sample_img: np.ndarray) -> np.ndarray:

    # Convert to float for accurate calculations
    template = template.astype(np.float64)
    overlap_mask = overlap_mask.astype(np.float64)
    sample_img = sample_img.astype(np.float64)
    
    # Inner function to compute SSD for each color channel
    def compute_channel_ssd(channel: int):
        masked_template = overlap_mask[:, :, channel] * template[:, :, channel]
        return (
            (masked_template ** 2).sum() 
            - 2 * cv2.filter2D(sample_img[:, :, channel], ddepth=-1, kernel=masked_template)
            + cv2.filter2D(sample_img[:, :, channel] ** 2, ddepth=-1, kernel=overlap_mask[:, :, channel])
        )

    # Compute SSD for each color channel
    ssd_blue = compute_channel_ssd(0)
    ssd_green = compute_channel_ssd(1)
    ssd_red = compute_channel_ssd(2)

    # Return the total SSD across all channels
    return ssd_blue + ssd_green + ssd_red



def select_low_cost_patch(cost_matrix: np.ndarray, tolerance: int) -> np.ndarray:
    # Find indices of the lowest cost values within the tolerance
    indices = np.argpartition(cost_matrix.ravel(), tolerance - 1)[:tolerance]
    lowest_cost_positions = np.column_stack(np.unravel_index(indices, cost_matrix.shape))
    
    # Randomly choose one of the patches with the lowest cost
    return random.choice(lowest_cost_positions)
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
   
    input_image = io.imread(input_image_file)
    target_image = io.imread(target_image_file)
    target_h = target_image.shape[1]  # Width
    target_w = target_image.shape[0]  # Height

    output = np.zeros((target_w, target_h, 3)) #This creates empty array for output image
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
                
                ssd_overlap = compute_patch_similarity(template, mask, input_image)
                ssd_overlap = ssd_overlap[half:-half, half:-half]  
                
                ssd_target = compute_patch_similarity(_target, np.ones((patch_size, patch_size, 3)), input_image)
                ssd_target = ssd_target[half:-half, half:-half] 
               
                ssd_prev = 0
                if n > 0:  # n is number of iteration
                    ssd_prev = compute_patch_similarity(synthesized, np.ones((patch_size, patch_size, 3)), input_image)
                    ssd_prev = ssd_prev[half:-half, half:-half] 
                
                ssd = (ssd_overlap + ssd_prev) * alpha + ssd_target * (1 - alpha)
                x, y = select_low_cost_patch(ssd, tol)

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



   