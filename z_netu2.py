# https://github.com/nayem-cosmic/FEM-Plate-MZC-Python/blob/master/stress_plate_mzc.py

import numpy as np

a = 1
b = 2
x = 3
y = 4

b_matrix = np.zeros( (3, 12) )

b_matrix[0][0] = -3*(x-x*y)/(4*a**2)
b_matrix[0][1] = -((3*a*x-3*a*x*y-a+a*y)/4)/a**2
b_matrix[0][2] = 0
b_matrix[0][3] = -3*(-x+x*y)/(4*a**2)
b_matrix[0][4] = -((3*a*x-3*a*x*y+a-a*y)/4)/a**2
b_matrix[0][5] = 0
b_matrix[0][6] = -3*(-x-x*y)/(4*a**2)
b_matrix[0][7] = -((3*a*x+3*a*x*y+a+a*y)/4)/a**2
b_matrix[0][8] = 0
b_matrix[0][9] = -3*(x+x*y)/(4*a**2)
b_matrix[0][10] = -((3*a*x+3*a*x*y-a-a*y)/4)/a**2
b_matrix[0][11] = 0
b_matrix[1][0] = -3*(y-x*y)/(4*b**2)
b_matrix[1][1] = 0
b_matrix[1][2] = -((3*b*y-3*b*x*y-b+b*x)/4)/b**2
b_matrix[1][3] = -3*(y+x*y)/(4*b**2)
b_matrix[1][4] = 0
b_matrix[1][5] = -((3*b*y+3*b*x*y-b-b*x)/4)/b**2
b_matrix[1][6] = -3*(-y-x*y)/(4*b**2)
b_matrix[1][7] = 0
b_matrix[1][8] = -((3*b*y+3*b*x*y+b+b*x)/4)/b**2
b_matrix[1][9] = -3*(-y+x*y)/(4*b**2)
b_matrix[1][10] = 0
b_matrix[1][11] = -((3*b*y-3*b*x*y+b-b*x)/4)/b**2
b_matrix[2][0] = -2*(1/2-3*x**2/8-3*y**2/8)/(a*b)
b_matrix[2][1] = -2*(-3/8*a*x**2+a*x/4+a/8)/(a*b)
b_matrix[2][2] = -2*(-3/8*b*y**2+b*y/4+b/8)/(a*b)
b_matrix[2][3] = -2*(-1/2+3*x**2/8+3*y**2/8)/(a*b)
b_matrix[2][4] = -2*(-3/8*a*x**2-a*x/4+a/8)/(a*b)
b_matrix[2][5] = -2*(3/8*b*y**2-b*y/4-b/8)/(a*b)
b_matrix[2][6] = -2*(1/2-3*x**2/8-3*y**2/8)/(a*b)
b_matrix[2][7] = -2*(3/8*a*x**2+a*x/4-a/8)/(a*b)
b_matrix[2][8] = -2*(3/8*b*y**2+b*y/4-b/8)/(a*b)
b_matrix[2][9] = -2*(-1/2+3*x**2/8+3*y**2/8)/(a*b)
b_matrix[2][10] = -2*(3/8*a*x**2-a*x/4-a/8)/(a*b)
b_matrix[2][11] = -2*(-3/8*b*y**2-b*y/4+b/8)/(a*b)

print(b_matrix)

# [[  6.75    6.     -0.     -6.75    7.5    -0.     11.25  -12.5    -0.
#   -11.25  -10.     -0.   ]
#  [  1.5    -0.      2.75   -3.     -0.     -5.5     3.     -0.     -6.5
#    -1.5    -0.      3.25 ]
#  [  8.875   2.5     9.75   -8.875   4.     -9.75    8.875  -4.    -13.75
#    -8.875  -2.5    13.75 ]]
