from abc import ABC, abstractmethod
from typing import Tuple, Any

import scipy as sp


class BaseMethod(ABC):

    def __init__(self, A: sp.sparse, max_iteration: int, tollerance: float, max_eigenvalues: int) -> None:
        self._A = A
        self._max_iteration = max_iteration
        self._tollerance = tollerance
        self._max_eigenvalues = max_eigenvalues

    @property
    def A(self) -> sp.sparse:
        return self._A

    @A.setter
    def A(self, value):
        self._A = value

    @property
    def max_iteration(self) -> int:
        return self._max_iteration

    @property
    def tollerance(self) -> float:
        return self._tollerance

    @property
    def max_eigenvalues(self) -> int:
        return self._max_eigenvalues

    @abstractmethod
    def compute(self) -> Tuple[Any, Any, Any, Any]:
        pass
