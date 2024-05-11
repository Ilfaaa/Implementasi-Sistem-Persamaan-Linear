import numpy as np

def lu_gauss_method(A, B):
    n = len(A)
    
    # Inisialisasi matriks L dan U
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    # Proses dekomposisi LU
    for k in range(n):
        L[k, k] = 1
        for j in range(k, n):
            U[k, j] = A[k, j] - np.dot(L[k, :k], U[:k, j])
        for i in range(k+1, n):
            L[i, k] = (A[i, k] - np.dot(L[i, :k], U[:k, k])) / U[k, k]
    
    # Solusi Ly = B
    Y = np.zeros(n)
    for i in range(n):
        Y[i] = B[i] - np.dot(L[i, :i], Y[:i])
    
    # Solusi Ux = y
    X = np.zeros(n)
    for i in range(n-1, -1, -1):
        X[i] = (Y[i] - np.dot(U[i, i+1:], X[i+1:])) / U[i, i]
    
    return X

# Input matriks koefisien A
A = np.array([[2, 1], [1, -1]])

# Input vektor hasil B
B = np.array([5, 0])

# Solusi sistem persamaan linear
X = lu_gauss_method(A, B)
print("Solusi menggunakan metode dekomposisi LU Gauss:", X)
