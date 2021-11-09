import scipy as sp
import numpy as np
from sympy import *

E  = 100 * 1e9
mi = 0.25
h  = 0.1
p  = 1000
a  = 0.5
b  = 0.5


x = Symbol("x")
y = Symbol("y")

r = x/a
s = y/b


D = E*(h**3) / (12 * (1 - (mi**2))) * np.array( [[1,  mi, 0       ],
                                                 [mi,  1, 0       ],
                                                 [0,   0, (1-mi)/2]] )

N = [0 for i in range(16)]
N[0]  = +1/16 * 1   * (+r-1)**2 * (-r-2)*(+s-1)**2 * (-s-2)
N[1]  = -1/16 * a   * (+r-1)**2 * (+r+1)*(+s-1)**2 * (-s-2)
N[2]  = -1/16 * b   * (+r-1)**2 * (-r-2)*(+s-1)**2 * (+s+1)
N[3]  = +1/16 * a*b * (+r-1)**2 * (+1+r)*(+s-1)**2 * (+s+1)
N[4]  = +1/16 * 1   * (+r+1)**2 * (+r-2)*(+s-1)**2 * (-s-2)
N[5]  = +1/16 * a   * (+r+1)**2 * (-r+1)*(+s-1)**2 * (-s-2)
N[6]  = -1/16 * b   * (+r+1)**2 * (+r-2)*(+s-1)**2 * (+s+1)
N[7]  = -1/16 * a*b * (+r+1)**2 * (+1-r)*(+s-1)**2 * (+s+1)
N[8]  = +1/16 * 1   * (+r+1)**2 * (+r-2)*(+s+1)**2 * (+s-2)
N[9]  = +1/16 * a   * (+r+1)**2 * (-r+1)*(+s+1)**2 * (+s-2)
N[10] = +1/16 * b   * (+r+1)**2 * (+r-2)*(+s+1)**2 * (-s+1)
N[11] = +1/16 * a*b * (+r+1)**2 * (+1-r)*(+s+1)**2 * (+1-s)
N[12] = +1/16 * 1   * (+r-1)**2 * (-r-2)*(+s+1)**2 * (+s-2)
N[13] = -1/16 * a   * (+r-1)**2 * (+r+1)*(+s+1)**2 * (+s-2)
N[14] = +1/16 * b   * (+r-1)**2 * (-r-2)*(+s+1)**2 * (-s+1)
N[15] = -1/16 * a*b * (+r-1)**2 * (+1+r)*(+s+1)**2 * (+1-s)

N_x  = [ 0 for i in range(len(N)) ]
N_y  = [ 0 for i in range(len(N)) ]
N_xy = [ 0 for i in range(len(N)) ]
F    = [ 0 for i in range(len(N)) ]
K    = np.zeros( (len(N), len(N)) )

cunt = 1

for i in range(16):
    N_x[i]  = N[i].diff(x,2)
    N_y[i]  = N[i].diff(y,2)
    N_xy[i] = N[i].diff(x,y)
    print("Kolo {}/16 done".format(cunt))
    cunt += 1

for i in range( len(N) ):
    for j in range( len(N) ):
        K[i][j] = K[i][j] + integrate( integrate(
        D[0][0]*N_x[i]*N_x[j] + D[1][1]*N_y[i]*N_y[j] +
        D[0][1]*( N_x[i]*N_y[j] + N_y[i]*N_x[j] ) +
        4*D[2][2]*N_xy[i]*N_xy[j], (x, -a, a) ), (y, -b, b) )
        print("Kolo i={}, j={} done".format(i, j))
    F[i] = integrate( integrate( p*N[i] , (x, -a, a) ), (y, -b, b) )


print(K)



# foo_der_x  = foo.diff(x,1)
# foo_der_y  = foo.diff(y,1)
# foo_der_xy = foo.diff(x,y)
#
# foo_inte_x = foo.integrate(x)
# foo_inte_y = foo.integrate(y)
#
# foo_def_inte_x = foo.integrate( (x, 1, 6) )
# foo_def_inte_y = foo.integrate( (y, 1, 6) )
#
# f_x = integrate(foo, (x, -1, 2))
# f_y = integrate(foo, (y, -3, 4))
# f_xy = integrate( (integrate(foo, (y, -3, 4))) , (x, -1, 2))
# f_yx = integrate( (integrate(foo, (x, -1, 2))) , (y, -3, 4))

# print(f_x, f_y)
# print(f_xy, f_yx)

# print(foo,"derivacia x", foo_der_x)
# print(foo,"derivacia y", foo_der_y)
# print(foo,"derivacia x y", foo_der_xy)
# print("-"*10)
# print(foo,"neurc integral x", foo_inte_x)
# print(foo,"neurc integral y", foo_inte_y)
# print("-"*10)
# print(foo,"urc integral x", foo_def_inte_x)
# print(foo,"urc integral y", foo_def_inte_y)
