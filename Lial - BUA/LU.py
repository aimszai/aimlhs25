import numpy as np

def read_csv():
    A = np.array([])
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

def LU(A):
    A = A.astype(float).copy()
    n = A.shape[0]
    L = np.eye(n)
    iter = 1
    while A[0,0] == 0:
        L[[0, iter]] = L[[iter, 0]]
        A = L @ A
        iter += 1
        if iter > n:
            raise ZeroDivisionError("Zero or near-zero pivot; use pivoting.")
    for k in range(n):
        pivot = A[k, k]
        for i in range(k + 1, n):
            L[i, k] = A[i, k] / pivot
            A[i, :] -= L[i, k] * A[k, :]
    U = A
    return L, U

def main():
    A = read_csv()
    L, U = LU(A)
    print(f"L:\n{L}\nU:\n{U}")

if __name__ == "__main__":
    main()