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

def markov_iter(A):
    tolerance = 1e-4
    x0 = np.zeros((A.shape[0], 1))
    x0[0] = 1
    max_iter = 1000
    #col_sums = A.sum(axis=0)
    for i in range(max_iter):
        xn = A @ x0
        err = np.linalg.norm(xn - x0)
        if err < tolerance:
            return xn
        print(f"Iteration {i+1}: Fehler = {err}")
        x0 = xn
    return xn

def markov(A):
    evals, evecs = np.linalg.eig(A)
    idx = np.argmin(np.abs(evals - 1.0)) # Index des Eigenwerts 1
    v = evecs[:, idx]
    v = v.real
    stationary_dist = v / np.sum(v)
    return stationary_dist

def main():
    A = read_csv()
    A_iter = markov_iter(A)
    A = markov(A)
    print(f"Matrix (Iterative Methode): \n{A_iter}")
    print(f"Matrix: \n{A}")

if __name__ == "__main__":
    main()