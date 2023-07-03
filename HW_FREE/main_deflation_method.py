import os
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import linalg

from eigenvalues.DeflationMethod import DeflationMethod
from utils.utils import calculate_adjacency_matrix, calculate_diagonal_matrix, calculate_laplacian_matrix


def main():
    np.random.seed(0)

    circle_path = os.path.join('dataset', 'Circle.csv')
    spiral_path = os.path.join('dataset', 'Spiral.csv')

    circle_df = pd.read_csv(circle_path, header=None, names=['x', 'y'])
    spiral_df = pd.read_csv(spiral_path, header=None, names=['x', 'y', 'cluster'])

    spiral_df = spiral_df[["x", "y"]]
    sigma = 1.0
    k_values = [5, 20]

    datasets = [circle_df, spiral_df]

    fig, ax = plt.subplots(len(k_values), len(datasets), figsize=(12, 6))
    fig.subplots_adjust(left=None, bottom=None, right=None,
                        top=0.82, wspace=0.40, hspace=None)

    for i, dataset in enumerate(datasets):
        for j, k in enumerate(k_values):
            adjacency_matrix = calculate_adjacency_matrix(dataset, sigma, k)
            degree_matrix = calculate_diagonal_matrix(adjacency_matrix)
            laplacian_matrix = calculate_laplacian_matrix(adjacency_matrix, degree_matrix)

            deflation_method = DeflationMethod(A=laplacian_matrix)
            eigenvalues, eigenvectors, _, _ = deflation_method.compute()

            theoric_eigenvalues = linalg.eigs(laplacian_matrix, which='SM', k=10)[0]
            theoric_eigenvalues = np.sort(theoric_eigenvalues)

            ax[j, i].plot(theoric_eigenvalues, '-o', label="Theoretical eigenvalues")
            ax[j, i].legend()
            ax[j, i].plot(np.abs(eigenvalues), '--+', label='Deflation power method')

            ax[j, i].set_ylabel(r'|$\lambda_i$|', fontweight='bold', fontsize=18)

            ax[j, i].legend()
            ax[j, i].set_xticks([s for s in range(10)])
            ax[j, i].set_xticklabels([f'$\lambda_{s + 1}$' for s in range(10)], fontsize=12)

    fig.suptitle('Deflation Power Method Assessment', fontweight='bold', fontsize=18)

    path_to_fig = os.path.join('figs', r'deflation_method_' + str(time.time()) + '.png')
    fig.savefig(path_to_fig, dpi=600)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
