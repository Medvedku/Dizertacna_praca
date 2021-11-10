import scipy as sp
import numpy as np
from sympy import *

np.set_printoptions(precision=7)

E  = 210 * 1e9
mi = 0.3
h  = 0.01
p  = 1000
a  = 0.5
b  = 0.5

D = E*h**3 / (12*(1-mi**2))

K_e = [0 for i in range(4)]

K_e[0] = b/(6*(a**3)) * np.array([
[6,     0,      0,   0,     0,     0,  0,   0,     0,  0,   0,     0],
[6*a,   8*a*a,  0,   0,     0,     0,  0,   0,     0,  0,   0,     0],
[0,     0,      0,   0,     0,     0,  0,   0,     0,  0,   0,     0],

[-6,   -6*a,    0,   6,     0,     0,  0,   0,     0,  0,   0,     0],
[ 6*a,  4*a*a,  0,  -6*a,   8*a*a, 0,  0,   0,     0,  0,   0,     0],
[ 0,    0,      0,   0,     0,     0,  0,   0,     0,  0,   0,     0],

[-3,   -3*a,    0,   3,    -3*a,   0,  6,   0,     0,  0,   0,     0],
[ 3*a,  2*a*a,  0,  -3*a,   4*a*a, 0, -6*a, 8*a*a, 0,  0,   0,     0],
[ 0,    0,      0,   0,     0,     0,  0,   0,     0,  0,   0,     0],

[3,     3*a,    0,  -3,     3*a,   0, -6,   6*a,   0,  6,   0,     0],
[3*a,   4*a*a,  0,  -3*a,   2*a*a, 0, -6*a, 4*a*a, 0,  6*a, 8*a*a, 0],
[0,     0,      0,   0,     0,     0,  0,   0,     0,  0,   0,     0],
])

K_e[1] = a/(6*(b**3)) * np.array([
[6,     0,      0,      0,     0,   0,      0,   0,     0,      0,   0,   0],
[0,     0,      0,      0,     0,   0,      0,   0,     0,      0,   0,   0],
[6*b,   0,      8*b*b,  0,     0,   0,      0,   0,     0,      0,   0,   0],

[3,     0,      3*b,    6,     0,   0,      0,   0,     0,      0,   0,   0],
[0,     0,      0,      0,     0,   0,      0,   0,     0,      0,   0,   0],
[3*b,   0,      4*b*b,  6*b,   0,   8*b*b,  0,   0,     0,      0,   0,   0],

[-3,    0,     -3*b,   -6,     0,  -6*b,    6,   0,     0,      0,   0,   0],
[ 0,    0,      0,      0,     0,   0,      0,   0,     0,      0,   0,   0],
[ 3*b,  0,      2*b*b,  6*b,   0,   4*b*b, -6*b, 0,     8*b*b,  0,   0,   0],

[-6,    0,     -6*b,   -3,     0,  -3*b,    3,   0,    -3*b,    6,   0,   0],
[ 0,    0,      0,      0,     0,   0,      0,   0,     0,      0,   0,   0],
[ 6*b,  0,      4*b*b,  3*b,   0,   2*b*b, -3*b, 0,     4*b*b, -6*b, 0,   8*b*b],
])

K_e[2] = mi/(2*a*b) * np.array([
[1,     0,      0,      0,     0,     0,      0,   0,     0,    0,   0,     0],
[a,     0,      0,      0,     0,     0,      0,   0,     0,    0,   0,     0],
[b,     2*a*b,  0,      0,     0,     0,      0,   0,     0,    0,   0,     0],

[-1,    0,     -b,      1,     0,     0,      0,   0,     0,    0,   0,     0],
[ 0,    0,      0,     -a,     0,     0,      0,   0,     0,    0,   0,     0],
[-b,    0,      0,      b,    -2*a*b, 0,      0,   0,     0,    0,   0,     0],

[1,     0,      0,     -1,     a,     0,      1,   0,     0,    0,   0,     0],
[0,     0,      0,      a,     0,     0,     -a,   0,     0,    0,   0,     0],
[0,     0,      0,      0,     0,     0,     -b,   2*a*b, 0,    0,   0,     0],

[-1,   -a,      0,      1,     0,     0,     -1,   0,     b,    1,   0,     0],
[-a,    0,      0,      0,     0,     0,      0,   0,     0,    a,   0,     0],
[ 0,    0,      0,      0,     0,     0,      0,   0,     0,   -b,  -2*a*b, 0],
])

K_e[3] = (1-mi)/(30*a*b) * np.array([
[21,     0,      0,      0,      0,      0,      0,    0,      0,      0,    0,     0],
[3*a,    8*a*a,  0,      0,      0,      0,      0,    0,      0,      0,    0,     0],
[3*b,    0,      8*b*b,  0,      0,      0,      0,    0,      0,      0,    0,     0],

[-21,   -3*a,   -3*b,    21,     0,      0,      0,    0,      0,      0,    0,     0],
[ 3*a,  -2*a*a,  0,     -3*a,    8*a*a,  0,      0,    0,      0,      0,    0,     0],
[-3*b,   0,     -8*b*b,  3*b,    0,      8*b*b,  0,    0,      0,      0,    0,     0],

[ 21,    3*a,    3*b,   -21,     3*a,   -3*b,    21,   0,      0,      0,    0,     0],
[-3*a,   2*a*a,  0,      3*a,   -8*a*a,  0,     -3*a,  8*a*a,  0,      0,    0,     0],
[-3*b,   0,      2*b*b,  3*b,    0,     -2*b*b, -3*b,  0,      8*b*b,  0,    0,     0],

[-21,   -3*a,   -3*b,    21,    -3*a,    3*b,   -21,   3*a,    3*b,    21,   0,     0],
[-3*a,  -2*a*a,  0,      3*a,    2*a*a,  0,     -3*a, -2*a*a,  0,      3*a,  8*a*a, 0],
[ 3*b,   0,     -8*b*b, -3*b,    0,      2*b*b,  3*b,  0,     -8*b*b, -3*b,  0,     8*b*b],
])

# apply symetry
for k in K_e:
    for i in range(12):
        for j in range(12):
            if j < i:
                pass
            else:
                k[i][j] = k[j][i]

# definition: stiffness matrix of element
K_elem = D * (K_e[0] + K_e[1] + K_e[2] + K_e[3])

F = [0 for i in range(12)]
F[0] = 1000

indexes = [i for i in range(12)]

deleto = [3,6,7,8,9]

K_elem = np.delete(K_elem, deleto, axis = 0)
K_elem = np.delete(K_elem, deleto, axis = 1)
F = np.delete(F, deleto, axis = 0)
indexes = np.delete(indexes, deleto, axis = 0)

delta = np.linalg.inv(K_elem)

r_tot = np.matmul(delta, F)

# for i in range(len(r_tot)):
#     print(indexes[i], r_tot[i])

spiel = True

class Node:
    num_of_nodes = 0
    def __init__(self, xz: list, alfa_ = 0):
        self.nd_id    = Node.num_of_nodes
        self.w        = Node.num_of_nodes*2 + 0
        self.fi_x     = Node.num_of_nodes*2 + 1
        self.fi_y     = Node.num_of_nodes*2 + 1

        Node.num_of_nodes += 1

LX = 3
LY = 5

nx = 4
ny = 4

coor_x = [0+i*(LX/nx) for i in range(nx+1)]
coor_y = [0+i*(LY/ny) for i in range(ny+1)]

print(coor_x)
print(coor_y)

Nodes_ = np.empty( (ny+1, nx+1), dtype = "int" )
cunt = 0
for i in range(ny+1):
    for j in range(nx+1):
        Nodes_[i][j] = cunt
        cunt += 1
print(Nodes_)

for i in range(ny+1):
    for j in range(nx+1):
        x_coor = coor_x[j]
        y_coor = coor_y[i]
        # print(int(Nodes_[i][j]), x_coor, y_coor)

Elems_ = np.empty( (ny, nx) )
cunt = 0
for i in range(ny):
    for j in range(nx):
        Elems_[i][j] = cunt
        print(cunt,
        "0:", Nodes_[i+1][j],
        "1:", Nodes_[i+1][j+1],
        "2:", Nodes_[i][j],
        "3:", Nodes_[i][j+1]
        )
        cunt += 1
