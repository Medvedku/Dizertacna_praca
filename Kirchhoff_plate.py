import scipy as sp
import numpy as np
from sympy import *
import sys
import timeit
import time
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv


np.set_printoptions(precision=15)

spiel = True

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
    a  = 0
    b  = 0
    E  = 0
    h  = 0
    mi = 0
    k  = np.zeros( (12,12) )

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
            Element.k = self.k_elem()
            self.k_el = Element.k
        else:
            self.k_el = Element.k

        Element.num_of_elements += 1

    def k_elem(self):
        a = self.a
        b = self.b
        E = self.E
        h = self.h
        mi = self.mi

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

        return K_elem

# Definition of construction

LX = 1
LY = 1

nx = 3
ny = 3

coor_x = [0+i*(LX/nx) for i in range(nx+1)]
coor_y = [0+i*(LY/ny) for i in range(ny+1)]


Nodes_ = np.empty( (ny+1, nx+1), dtype = "int" )
cunt = 0
for i in range(ny+1):
    for j in range(nx+1):
        Nodes_[i][j] = cunt
        cunt += 1
print(Nodes_)
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
        n2 = Nodes_[i][j]
        n3 = Nodes_[i][j+1]
        l_elems.append( Element(l_nodes[n0], l_nodes[n1], l_nodes[n2], l_nodes[n3], h = 0.01) )

# for i in l_elems:
#     print(i.el_id, i.vec)

# Assembly of Global stiffness B_matrix
n_e       = l_elems[-1].el_id + 1
n_n       = l_nodes[-1].nd_id + 1
dofs      = l_nodes[-1].fi_y+1
code_nums = list(range(dofs))
load      = [0 for i in range(dofs)]
load[0]   = 1

spiel = True
if spiel:
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

### Supports in corners of slab
boundary = []
for i in l_nodes:
    if i.co_x == 0 and i.co_y == LY:
        boundary.append(i.w)
    if i.co_x == LX and i.co_y == 0:
        boundary.append(i.w)
    if i.co_x == LX and i.co_y == LY:
        boundary.append(i.w)
print(boundary)
deleto = boundary
K_gl = np.delete(K_gl, deleto, axis = 0)
K_gl = np.delete(K_gl, deleto, axis = 1)
load = np.delete(load, deleto, axis = 0)
code_nums = np.delete(code_nums, deleto, axis = 0)

delta = np.linalg.inv(K_gl)

r_tot = np.matmul(delta, load)

print(r_tot[0])
