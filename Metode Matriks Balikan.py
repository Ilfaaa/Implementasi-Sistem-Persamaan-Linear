import numpy as np

def inverse_matrix_method(A, B):
    A_inv = np.linalg.inv(A)
    X = np.dot(A_inv, B)
    return X

# Testing
A = np.array([[2, 1], [1, -1]])
B = np.array([5, 0])
X = inverse_matrix_method(A, B)
print("Solusi menggunakan metode matriks balikan:", X)
