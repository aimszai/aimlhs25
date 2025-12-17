import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def read_csv():
    A = np.array([])
    b = np.array([])
    x = open('prüfungsvorbereitung/matrix.csv','r')
    #x = open('fisher.csv','r')
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

def mp_inverse(A):
    U, S, Vt = la.svd(A, full_matrices=False)
    S_inv = np.zeros((Vt.shape[0], U.shape[0]))
    for i in range(len(S)):
        if S[i] != 0:
            S_inv[i, i] = 1 / S[i]
    A_inv = Vt.T @ S_inv @ U.T
    return A_inv

def main():
    A = read_csv()
    A_inv = mp_inverse(A)
    print(f"Matrix: \n{A}")

if __name__ == "__main__":
    main()