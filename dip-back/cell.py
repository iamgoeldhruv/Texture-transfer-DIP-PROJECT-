from typing import Tuple, Optional
import numpy as np

class Cell:
    def __init__(self, i: int, j: int, cost: np.ndarray, before: Optional['Cell']):
        self.i = i
        self.j = j
        self.cost = cost
        self.before = before

    @property
    def point(self) -> Tuple[int, int]:
        return (self.i, self.j)
