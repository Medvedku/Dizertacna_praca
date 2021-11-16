import scipy as sp
import numpy as np
from sympy import *
import sys
import timeit
import time
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv


np.set_printoptions(precision=1)

spiel = True

def check_symmetric(a, rtol=1e-05, atol=1e-05):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

class Node:
    num_of_nodes = 0
    def __init__(self, x, y):
        self.nd_id    = Node.num_of_nodes
        self.w        = Node.num_of_nodes*3 + 0
        self.fi_x     = Node.num_of_nodes*3 + 1
        self.fi_y     = Node.num_of_nodes*3 + 2
        self.co_x     = x
        self.co_y     = y

        Node.num_of_nodes += 1

class Element:
    num_of_elements = 0
    a  = None
    b  = None
    E  = None
    h  = None
    mi = None
    k  = None
    d_mat  = None
    b_mat  = None

    def __init__(self, n0, n1, n2, n3, E = 210*1e9, mi = 0.3, h = 0.01):
        self.Flag   = False

        self.el_id  = Element.num_of_elements
        self.n_0    = n0.nd_id
        self.n_1    = n1.nd_id
        self.n_2    = n2.nd_id
        self.n_3    = n3.nd_id
        self.a      = abs(n0.co_x - n1.co_x)/2
        self.b      = abs(n1.co_y - n2.co_y)/2
        self.E      = E
        self.h      = h
        self.mi     = mi
        self.vec    = [ n0.w, n0.fi_x, n0.fi_y,
                        n1.w, n1.fi_x, n1.fi_y,
                        n2.w, n2.fi_x, n2.fi_y,
                        n3.w, n3.fi_x, n3.fi_y, ]

        if Element.num_of_elements == 0:
            Element.a  = self.a
            Element.b  = self.b
            Element.E  = self.E
            Element.h  = self.h
            Element.mi = self.mi
            if spiel:
                print("First element created", self.el_id)
            self.Flag   = True

        if abs((self.a / Element.a)-1)*100 > 1:
            if spiel:
                print("a changed, from: {} to: {}, element: {}".format( Element.a,self.a, self.el_id  ))
            Element.a  = self.a
            self.Flag   = True

        if abs((self.b / Element.b)-1)*100 > 1:
            if spiel:
                print("b changed, from: {} to: {}, element: {}".format( Element.b,self.b, self.el_id  ))
            Element.b  = self.b
            self.Flag   = True

        if abs((self.E / Element.E)-1)*100 > 1:
            if spiel:
                print("E changed, from: {} to: {}, element: {}".format( Element.E,self.E, self.el_id  ))
            Element.E  = self.E
            self.Flag   = True

        if abs((self.h / Element.h)-1)*100 > 1:
            if spiel:
                print("h changed, from: {} to: {}, element: {}".format( Element.h,self.h, self.el_id  ))
            Element.h  = self.h
            self.Flag   = True

        if abs((self.mi / Element.mi)-1)*100 > 1:
            if spiel:
                print("mi changed, from: {} to: {}, element: {}".format( Element.mi,self.mi, self.el_id  ))
            Element.mi  = self.mi
            self.Flag   = True

        if self.Flag:
            Element.k, Element.b_mat, Element.d_mat = self.k_elem()
            self.k_el  = Element.k
            self.b_mat = Element.b_mat
            self.d_mat = Element.d_mat
        else:
            self.k_el  = Element.k
            self.b_mat = Element.b_mat
            self.d_mat = Element.d_mat

        Element.num_of_elements += 1

    def k_elem(self):
        a = self.a
        b = self.b
        E = self.E
        h = self.h
        mi = self.mi

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
            b_mat_b = self.B_matrix(a, b, x_g, y_g)
            if i == 0:
                b_mat = b_mat_b
            else:
                b_mat = np.concatenate( ( b_mat, b_mat_b) )

        K_elem = (np.transpose(b_mat)@d_mat@b_mat)*a*b

        return K_elem, b_mat, d_mat

    def B_matrix(self, a, b, x, y):
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

# Definition of construction

LX = 1
LY = 1

nx = 2
ny = 2

coor_x = [0+i*(LX/nx) for i in range(nx+1)]
coor_y = [0+i*(LY/ny) for i in range(ny+1)]

Nodes_ = np.empty( (ny+1, nx+1), dtype = "int" )
cunt = 0
for i in range(ny+1):
    for j in range(nx+1):
        Nodes_[i][j] = cunt
        cunt += 1

l_nodes = []
for i in range(ny+1):
    for j in range(nx+1):
        x_coor = coor_x[j]
        y_coor = coor_y[i]
        l_nodes.append(Node(x_coor,y_coor))

l_elems = []
for i in range(ny):
    for j in range(nx):
        n0 = Nodes_[i+1][j]
        n1 = Nodes_[i+1][j+1]
        n2 = Nodes_[i][j+1]
        n3 = Nodes_[i][j]
        l_elems.append( Element(l_nodes[n0], l_nodes[n1], l_nodes[n2], l_nodes[n3], h = 0.01) )

# Assembly of Global stiffness B_matrix
n_e       = len(l_elems)
n_n       = len(l_nodes)
dofs      = n_n*3
code_nums = list(range(dofs))

load      = [0 for i in range(dofs)]
load[0]   = 1000

boundary = []
for i in l_nodes:
    if i.co_x == 0 and i.co_y == LY:
        boundary.append(i.w)
    if i.co_x == LX and i.co_y == 0:
        boundary.append(i.w)
    if i.co_x == LX and i.co_y == LY:
        boundary.append(i.w)

deleto = boundary



spiel = False
cunt = 0
step = 10
c_spiel = [i*((n_e*12*12)//10) for i in range(10)]

K_gl = np.zeros( (dofs, dofs) )
for i in l_elems:
    for j in range(12):                 # because element matrix is 12x12 shape
        for k in range(12):             # because element matrix is 12x12 shape
            K_gl[i.vec[j],i.vec[k]] += i.k_el[j,k]
            if spiel:
                if cunt in c_spiel:
                    print("Assembling global stiffness matrix: {}%".format(step))
                    step += 10
            cunt += 1
# np.savetxt("2elems_full.csv",K_gl, fmt = "%.2e")

K_gl = np.delete(K_gl, deleto, axis = 0)
K_gl = np.delete(K_gl, deleto, axis = 1)
load = np.delete(load, deleto, axis = 0)

code_nums = np.delete(code_nums, deleto, axis = 0)

delta = np.linalg.inv(K_gl)
r_tot = np.matmul(delta, load)

results = [ "{}: {} (in deg:{})".format(code_nums[i],r_tot[i]*1000,np.rad2deg(r_tot[i])) for i in range(len(code_nums)) ]
for i in results:
    print(i)
print(check_symmetric(K_gl))