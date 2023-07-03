from typing import Tuple, Any

import numpy as np
import scipy as sp
import scipy.sparse.linalg as spla

from eigenvalues.BaseMethod import BaseMethod


class InversePowerMethod(BaseMethod):

    def __init__(self, A: sp.sparse, max_iteration: int = 50, tollerance: float = 1e-15,
                 max_eigenvalues: int = 0) -> None:
        super().__init__(A, max_iteration, tollerance, max_eigenvalues)

    def compute(self) -> Tuple[Any, Any, Any, Any]:
        """
            Computes the smallest eigenvalue and corresponding eigenvector of a
            sparse matrix using the inverse power method.

            Inputs:
            - A: Sparse matrix (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix,
            or compatible sparse format)
            - max_iteration: Maximum number of iterations (default: 50)
            - tollerance: Tolerance for convergence (default: 1e-15)

            Outputs:
            - smallest_eigenvalue: Computed smallest eigenvalue
            - eigenvector: Normalized eigenvector corresponding
                to the smallest eigenvalue
            - eigenvector_iterations: List of eigenvector
                approximations at each iteration
            - eigenvalue_iterations: List of eigenvalues computed at each iteration
            """

        v_old = np.zeros(self.A.shape[0]) + 1e-7 * np.random.rand(self.A.shape[0])
        v_old = v_old / np.linalg.norm(v_old)

        lambda_1_inv_new = 1

        LU = spla.splu(self.A)  # LU factorization of the sparse matrix A

        eigenvector_iterations = [v_old.copy()]  # List to store eigenvector
        # approximations at each iteration
        eigenvalue_iterations = []  # List to store
        # eigenvalues computed at each iteration

        for k in range(self.max_iteration):
            # Update the eigenvalue
            lambda_1_inv_old = lambda_1_inv_new

            # Solve the linear system using LU factorization
            v_new = LU.solve(v_old)

            # Compute the eigenvalue
            lambda_1_inv_new = v_old.T.dot(v_new)

            # Normalize the current eigenvector approximation
            v_old = v_new / np.linalg.norm(v_new)

            # Store the current eigenvector approximation and eigenvalue
            eigenvector_iterations.append(v_old)
            eigenvalue_iterations.append(1 / lambda_1_inv_new)

            if abs(1 / lambda_1_inv_new - 1 / lambda_1_inv_old) < self.tollerance:  # Check for convergence
                break

        return 1 / lambda_1_inv_new, v_old, eigenvector_iterations, eigenvalue_iterations
