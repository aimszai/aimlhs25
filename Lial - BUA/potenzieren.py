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

def potenzierung(A, p):
    p = int(p)
    eigenwerte, eigenvektoren = np.linalg.eig(A)
    #print(f"Eigenwerte: \n{eigenwerte}")
    for e in range(len(eigenwerte)):
        eigenwerte[e] = eigenwerte[e] ** p
    Lambda = np.diag(eigenwerte)
    #print(f"Lambda: \n{Lambda}")

    V = np.column_stack((eigenvektoren[:,0], eigenvektoren[:,1]))

    result = V @ Lambda @ np.round(np.linalg.inv(V))
    np.round(result, out=result)  

    return result


def main():
    A = read_csv()
    p = input("Geben Sie die Potenz ein: ")
    A = potenzierung(A,p)
    #np.set_printoptions(suppress=True)
    print(f"Matrix: \n{A}")

if __name__ == "__main__":
    main()