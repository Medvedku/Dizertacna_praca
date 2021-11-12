import numpy as np


a = 0.5
b = 0.5
E = 200000000000
h = 0.01
mi = 0.3

D = (E*(h**3)) / (12*(1-mi**2))

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

deleto = [3,6,9]
print(deleto)

load = [0 for i in range(12)]
load[0] = 1000
K_elem = np.delete(K_elem, deleto, axis = 0)
K_elem = np.delete(K_elem, deleto, axis = 1)
load = np.delete(load, deleto, axis = 0)

delta = np.linalg.inv(K_elem)

r_tot = np.matmul(delta, load)


print(r_tot)