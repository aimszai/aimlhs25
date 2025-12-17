import numpy as np
import matplotlib.pyplot as plt

def read_csv():
    A = np.array([])
    b = np.array([])
    x = open('prüfungsvorbereitung/matrix.csv','r')
    lines = x.readlines()
    x.close()
    for x in lines[1:]:
        key, value = x.strip().split(';')
        lines = value.split(' ')
        lines = [float(i) for i in lines]
        if key == 'A':
            row = np.asarray(lines, dtype=float)
            if A.size == 0:
                A = row[np.newaxis, :]
            else:
                A = np.vstack((A, row))
    return A

def singular_value_decomposition(A):
    eigval, eigvec = np.linalg.eig(A.T @ A)
    sorted_indices = np.argsort(eigval)[::-1]
    eigval = eigval[sorted_indices]
    eigvec = eigvec[:, sorted_indices]
    sqrt_eigval = np.sqrt(eigval)
    sigma = np.diag(sqrt_eigval)
    U = A @ eigvec @ np.linalg.inv(sigma)
    np.round(U, decimals=6, out=U)
    np.round(sigma, decimals=6, out=sigma)
    np.round(eigvec, decimals=6, out=eigvec)
    return U, eigvec, sigma

def main():
    A = read_csv()
    U, V, sigma = singular_value_decomposition(A)

    print(f"Matrix: \n{A}")
    print(f"U: \n{U}")
    print(f"V: \n{V}")
    print(f"Sigma: \n{sigma}")
    print(f"Reconstructed A (U * Sigma * V^T): \n{np.round(U @ sigma @ V.T)}")

if __name__ == "__main__":
    main()