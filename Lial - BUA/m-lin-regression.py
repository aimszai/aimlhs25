import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la

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
            
    A =np.hstack((np.ones((A.shape[0],1)), A))
    return A,b

def linear_regression(X,y):
    beta_hat = la.inv(X.T @ X) @ X.T @ y
    y_hat = X @ beta_hat # predicted y values
    return beta_hat, y_hat

def visualisation(beta_hat, y, y_hat, X):

    x = np.linspace(-10,10,100)
    plt.figure(figsize=(10, 6))
    plt.scatter(np.arange(1, y.shape[0] + 1), y, label='Actual y', color='blue', marker='o', s=100)
    plt.plot(np.arange(1, y.shape[0] + 1), y_hat, label='Fitted y', color='red', linewidth=2)
    plt.xlabel('Observation Index')
    plt.ylabel('y')
    plt.title(f'Linear Regression: y = {beta_hat[0]:.4f} + {beta_hat[1]:.4f}*x1 + {beta_hat[2]:.4f}*x2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
def main():
    X, y = read_csv()
    beta_hat, y_hat = linear_regression(X, y)
    print(f"Estimated coefficients (beta_hat):\n{beta_hat}\ny_hat:\n{y_hat}")
    visualisation(beta_hat, y, y_hat,X)

if __name__ == "__main__":
    main()