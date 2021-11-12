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
a = Symbol("a")
b = Symbol("b")

D = E*(h**3) / (12 * (1 - (mi**2))) * np.array( [[1,  mi, 0       ],
                                                 [mi,  1, 0       ],
                                                 [0,   0, (1-mi)/2]] )

r = x / a
s = y / b

N = [0 for i in range(16)]

r_i = -1; s_i = -1
N0   = 1/8*(1 + (r_i)*r)*(1 + (s_i)*s)*(2 + (r_i)*r)+(s_i)*s+(r**2)+(s**2)
N0_  = 1/8*a*((r**2) - 1)*(r+r_i)*(1+(s_i*s))

print(N0.diff(x,1))
print(N0_.simplify())
