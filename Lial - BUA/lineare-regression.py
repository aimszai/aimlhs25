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
        elif key == 'b':
            b = np.asarray(lines, dtype=float)
    return A,b

def linear_regression(A, b):
    x_hat = np.linalg.inv(A.T @ A) @ A.T @ b
    print(x_hat)
    C,D = x_hat
    return C,D

def visualisation(C, D, b, A):
    x = np.linspace(-10,10,100)
    y = C * x + D
    plt.plot(x, y, label=f'y = {C:.2f}x + {D:.2f}')
    # plot the original data points using the first column of A as x-values
    plt.plot(A[:,0], b, 'ro', label='Datenpunkte')
    plt.title('Lineare Regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(0, color='black',linewidth=0.5, ls='--')
    plt.axvline(0, color='black',linewidth=0.5, ls='--')
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.legend()
    plt.show()

def main():
    A,b = read_csv()
    C,D = linear_regression(A, b)
    visualisation(C, D, b, A)
    print(f"A:\n{A}\nb:\n{b}")
    print(f"A:\n{A}\nb:\n{b}")

if __name__ == "__main__":
    main()