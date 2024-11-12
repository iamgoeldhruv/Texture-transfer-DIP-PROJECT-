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