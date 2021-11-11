import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv

A = np.array( [ [ 2,  0,  0,  0,  0,  0],
                [ 0,  9, -3, -1,  0,  0],
                [ 0,  0,  5,  0,  0,  0],
                [-2,  0,  0, -7, -1,  0],
                [-1,  0,  0, -5,  1, -3],
                [-1, -2,  0,  0,  0,  6]
                ] )

# S1 = csr_matrix(A)
# print("Sprase CSR matrix: \n", S1)

# S1_inv = inv(S1)
# print("Inverse CSR matrix: \n", S1_inv)

S2 = csc_matrix(A)
# print("Sprase CSC matrix: \n", S2)

S2_inv = inv(S2)
# print("Inverse CSC matrix: \n", S2_inv)

# B1 = S1_inv.todense()

B2 = S2_inv.todense()

# print(B1)
print(10*"-")
print(B2)
