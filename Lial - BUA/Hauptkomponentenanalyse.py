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

def analyse(A_tilde):
    m = A_tilde.shape[0]
    M = np.eye (m) - np.ones ((m, m)) / m
    A = M @ A_tilde
    C = A.T @ A / (m - 1)
    EW, EV = la.eig (C)
    idx = np.abs(EW).argsort()[::-1]
    EW = EW[idx]
    EV = EV[:,idx]
    A @= EV
    plt.plot(np.cumsum(EW) / np.sum(EW), marker='o')
    plt.title('Kumulative Varianz erklärt durch Hauptkomponenten')
    plt.xlabel('Anzahl Hauptkomponenten')
    plt.ylabel('Kumulative Varianz')
    plt.grid()
    plt.show()
    return A

def main():
    A = read_csv()
    analyse(A)
    print(f"Matrix: \n{A}")

if __name__ == "__main__":
    main()