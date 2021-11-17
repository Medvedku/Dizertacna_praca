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
a = Symbol("a")
b = Symbol("b")

D = E*(h**3) / (12 * (1 - (mi**2))) * np.array( [[1,  mi, 0       ],
                                                 [mi,  1, 0       ],
                                                 [0,   0, (1-mi)/2]] )


chi = +1
eta = -1

N_0_0     = -3*(x-x*y)/(4*a**2)
N_0_0_d   = (-1/(4*a**2)) * (3*chi*x + 3*chi*eta*x*y)

N_d_0_0   = -((3*a*x-3*a*x*y-a+a*y)/4)/a**2
N_d_0_0_d = (1/(4*a)) * (3*x + chi*eta*y + 3*eta*x*y + chi)


print(N_0_0.simplify())
print(N_0_0_d.simplify())
print(5*"-")
print(N_d_0_0.simplify())
print(N_d_0_0_d.simplify())
print(10*"-")
print(N_0_0.diff(x,2))
print(N_0_0.diff(y,2))
print(N_0_0.diff(x,y))

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
