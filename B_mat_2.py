# https://github.com/nayem-cosmic/FEM-Plate-MZC-Python/blob/master/stress_plate_mzc.py

import numpy as np

np.set_printoptions(precision=1)


def B_matrix(a, b, x, y):
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
    return b_matrix

E  = 200000000000
mi = 0.3
h  = 0.01
p  = 1000
a  = 0.25
b  = 0.5


d_mat = np.zeros( (12,12) )
D_con = E*h**3/(12*(1-mi**2))
d_mat[0][0], d_mat[1][1], d_mat[3][3], d_mat[4][4]   = D_con * 1, D_con * 1, D_con * 1, D_con * 1
d_mat[6][6], d_mat[7][7], d_mat[9][9], d_mat[10][10] = D_con * 1, D_con * 1, D_con * 1, D_con * 1

d_mat[0][1], d_mat[1][0], d_mat[3][4], d_mat[4][3]   = D_con * mi, D_con * mi, D_con * mi, D_con * mi
d_mat[6][7], d_mat[7][6], d_mat[9][10], d_mat[10][9] = D_con * mi, D_con * mi, D_con * mi, D_con * mi

d_mat[2][2], d_mat[5][5], d_mat[8][8], d_mat[11][11] = D_con * (1-mi)/2, D_con * (1-mi)/2, D_con * (1-mi)/2, D_con * (1-mi)/2

gauss_x=[-1/3**0.5, +1/3**0.5, +1/3**0.5, -1/3**0.5]
gauss_y=[-1/3**0.5, -1/3**0.5, +1/3**0.5, +1/3**0.5]

b_mat = None
for i in range(4):
    x_g = gauss_x[i]
    y_g = gauss_y[i]
    b_mat_b = B_matrix(a, b, x_g, y_g)
    if i == 0:
        b_mat = b_mat_b
    else:
        b_mat = np.concatenate( ( b_mat, b_mat_b) )

k_elem = (np.transpose(b_mat)@d_mat@b_mat)*a*b

print(k_elem)

indexes = [i for i in range(12)]
load    = [0 for i in range(12)]
load[0] = 1000
deleto  = [3,6,9]

k_elem = np.delete(k_elem, deleto, axis = 0)
k_elem = np.delete(k_elem, deleto, axis = 1)
load = np.delete(load, deleto, axis = 0)
indexes = np.delete(indexes, deleto, axis = 0)

delta = np.linalg.inv(k_elem)
r_tot = np.matmul( delta, load )


print(r_tot)
results = [ "{}: {} (in deg:{})".format(indexes[i],r_tot[i]*1000,np.rad2deg(r_tot[i])) for i in range(len(indexes)) ]
for i in results:
    print(i)
