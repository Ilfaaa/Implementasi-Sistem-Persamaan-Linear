import numpy as np

def crout_method(A, B):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for j in range(n):
        U[j, j] = 1
        for i in range(j, n):
            sum1 = sum(U[k, j] * L[i, k] for k in range(j))
            L[i, j] = A[i, j] - sum1
        for i in range(j, n):
            sum2 = sum(U[k, j] * L[i, k] for k in range(j))
            U[j, i] = (A[j, i] - sum2) / L[j, j]

    Y = np.linalg.solve(L, B)
    X = np.linalg.solve(U, Y)
    return X

# Testing
A = np.array([[2, 1], [1, -1]])
B = np.array([5, 0])
X = crout_method(A, B)
print("Solusi menggunakan metode dekomposisi Crout:", X)
