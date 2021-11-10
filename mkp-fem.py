import scipy as sp
import numpy as np
from sympy import *

np.set_printoptions(precision=3)

E  = 210 * 1e9
mi = 0.3
h  = 0.01
p  = 1000
a  = 0.5
b  = 0.5

x = Symbol("x")
y = Symbol("y")

D = E*(h**3) / (12 * (1 - (mi**2))) * np.array( [[1,  mi, 0       ],
                                                 [mi,  1, 0       ],
                                                 [0,   0, (1-mi)/2]] )

r = x / a
s = y / b

N = [0 for i in range(16)]

N_0  = +1/16 * 1   * (+r-1)**2 * (-r-2)*(+s-1)**2 * (-s-2)
N_5  = +1/16 * a   * (+r+1)**2 * (-r+1)*(+s-1)**2 * (-s-2)
N_7  = -1/16 * a*b * (+r+1)**2 * (+1-r)*(+s-1)**2 * (+s+1)
N_14 = +1/16 * b   * (+r-1)**2 * (-r-2)*(+s+1)**2 * (-s+1)
N_15 = -1/16 * a*b * (+r-1)**2 * (+1+r)*(+s+1)**2 * (+1-s)

N = [N_0, N_5, N_7, N_14, N_15]

N_x  = [ 0 for i in range(len(N)) ]
N_y  = [ 0 for i in range(len(N)) ]
N_xy = [ 0 for i in range(len(N)) ]
F    = [ 0 for i in range(len(N)) ]
K    = np.zeros( (len(N), len(N)) )

# K    = np.array(
# [[108226.331,  18923.623,   1566.387,  18923.623,   1566.387],
#  [ 18923.623,  17443.137,   2935.076,  -3100.28,    -998.319],
#  [  1566.387,   2935.076,   1035.128,   -998.319,   -231.601],
#  [ 18923.623,  -3100.28,    -998.319,  17443.137,   2935.076],
#  [  1566.387,   -998.319,   -231.601,   2935.076,   1035.128]])

for i in range(len(N)):
    N_x[i]  = N[i].diff(x,2)
    N_y[i]  = N[i].diff(y,2)
    N_xy[i] = N[i].diff(x,y)

for i in range( len(N) ):
    for j in range( len(N) ):
        pass
        K[i][j] = K[i][j] + integrate( integrate(
        D[0][0]*N_x[i]*N_x[j] + D[1][1]*N_y[i]*N_y[j] +
        D[0][1]*( N_x[i]*N_y[j] + N_y[i]*N_x[j] ) +
        4*D[2][2]*N_xy[i]*N_xy[j], (x, -a, a) ), (y, -b, b) )
    F[i] = integrate( integrate( p*N[i] , (x, -a, a) ), (y, -b, b) )

delta = np.linalg.inv(K)
print( np.array(F) )
print( np.array(K) )
print(delta)

u = np.matmul(delta, F)
print(u)
