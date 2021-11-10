import scipy as sp
import numpy as np
from sympy import *

E  = 100 * 1e9
mi = 0.25
h  = 0.1

x = Symbol("x")
y = Symbol("y")
a = Symbol("a")
b = Symbol("b")

r = x/a
s = y/b

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

print(N[0].diff(x,2))
print(N[0].diff(y,2))
print(N[0].diff(x,y))