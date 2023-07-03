from typing import Tuple, Any

import numpy as np
import scipy as sp
import scipy.sparse.linalg as spla

from eigenvalues.BaseMethod import BaseMethod
from eigenvalues.InversePowerMethod import InversePowerMethod


class DeflationMethod(BaseMethod):

    def __init__(self, A: sp.sparse, max_iteration: int = 100, tollerance: float = 1e-25,
                 max_eigenvalues: int = 10) -> None:
        super().__init__(A, max_iteration, tollerance, max_eigenvalues)

    def _inverse_power_method(self):
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
        """
        v_old = np.ones((self.A.shape[0],)) + \
                1e-7 * np.random.rand(self.A.shape[0])  # Initial guess for eigenvector (!=0)

        v_old = v_old / np.linalg.norm(v_old)

        lambda_1_inv_new = 1

        LU = spla.splu(self.A)  # LU factorization of the sparse matrix A

        for k in range(self.max_iteration):

            # Update the eigenvalue
            lambda_1_inv_old = lambda_1_inv_new

            # Solve the linear system v_new = A^(-1) v_old using LU factorization
            v_new = LU.solve(v_old)

            # Compute the eigenvalue
            lambda_1_inv_new = v_old.T.dot(v_new)

            # Normalize the current eigenvector approximation
            v_old = v_new / np.linalg.norm(v_new)

            if abs(lambda_1_inv_new - lambda_1_inv_old) < self.tollerance:  # Check for convergence
                break

        return 1 / lambda_1_inv_new, v_old

    def compute(self) -> Tuple[Any, Any, Any, Any]:
        """
            Computes the M smallest eigenvalues and corresponding eigenvectors of a 
            sparse matrix using the deflation method.

            Inputs:
            - A: Sparse matrix (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix, 
            or compatible sparse format)
            - max_iteration: Maximum number of iterations for the inverse power 
            method (default: 50)
            - tollerance: Tolerance for convergence of the inverse power method 
            (default: 1e-15)
            - max_eigenvalues: Number of smallest eigenvalues to compute

            Outputs:
            - eigenvalues: Array of the M smallest eigenvalues
            - eigenvectors: Array of the M corresponding eigenvectors
            """

        eigenvalues = np.zeros(self.max_eigenvalues)
        eigenvectors = np.zeros((self.A.shape[0], self.max_eigenvalues))

        for i in range(self.max_eigenvalues):
            inverse_power_method = InversePowerMethod(A=self.A, max_iteration=self.max_iteration, tollerance=self.tollerance)
            eigenvalue, eigenvector, _, _ = inverse_power_method.compute()  # Compute smallest eigenvalue and
            # eigenvector

            eigenvalues[i] = eigenvalue  # Store eigenvalue
            eigenvectors[:, i] = eigenvector  # Store eigenvector

            # Deflation step: Update A by removing the contribution of the computed eigenvector
            self.A = self.A - eigenvalue / eigenvalues[0] * np.outer(eigenvector, eigenvector)

        return eigenvalues, eigenvectors, None, None
