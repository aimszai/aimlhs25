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

def check_det(A):
    det = np.linalg.det(A)
    if det == 0:
        print("Die Matrix ist singulär (Determinante ist 0).")
    else:
        print(f"Die Determinante der Matrix ist: {det}")
        exit()

def main():
    A = read_csv()
    eigenwerte, eigenvektoren = np.linalg.eig(A)
    print(f"Eigenwert: {eigenwerte[0]}\nEigenvektor: {eigenvektoren[:,0]}")
    print(f"Eigenwert: {eigenwerte[1]}\nEigenvektor: {eigenvektoren[:,1]}")

if __name__ == "__main__":
    main()