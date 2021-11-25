import numpy as np

np.set_printoptions(precision=1)

spiel = 1

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
        self.def_z    = 0

        Node.num_of_nodes += 1

    def apply_deformation(self, defm):
        self.def_z    -= defm


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

    aa = 1 + 3**(1/2)
    bb = 1 - 3**(1/2)
    mstress = (1/4) * np.array([[aa*aa, aa*bb, bb*bb, aa*bb],
                                [aa*bb, aa*aa, aa*bb, bb*bb],
                                [bb*bb, aa*bb, aa*aa, aa*bb],
                                [aa*bb, bb*bb, aa*bb, aa*aa] ])

    def __init__(self, n0, n1, n2, n3, E = 200*1e9, mi = 0.3, h = 0.01):
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

        self.e_loads = []

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
        b_matrix = []
        for i in range(4):
            x_g = gauss_x[i]
            y_g = gauss_y[i]
            b_mat_b = self.B_matrix(a, b, x_g, y_g)
            if i == 0:
                b_mat = b_mat_b
                b_matrix.append(b_mat_b)
            else:
                b_mat = np.concatenate( ( b_mat, b_mat_b) )
                b_matrix.append(b_mat_b)

        K_elem = (np.transpose(b_mat)@d_mat@b_mat)*a*b

        return K_elem, b_matrix, d_mat

    def B_matrix(self, a, b, x, y):
        b_matrix = np.zeros( (3, 12) )
        b_matrix[0][0]  = -3*(x-x*y)/(4*a**2)
        b_matrix[0][1]  = -((3*a*x-3*a*x*y-a+a*y)/4)/a**2
        b_matrix[0][2]  = 0
        b_matrix[0][3]  = -3*(-x+x*y)/(4*a**2)
        b_matrix[0][4]  = -((3*a*x-3*a*x*y+a-a*y)/4)/a**2
        b_matrix[0][5]  = 0
        b_matrix[0][6]  = -3*(-x-x*y)/(4*a**2)
        b_matrix[0][7]  = -((3*a*x+3*a*x*y+a+a*y)/4)/a**2
        b_matrix[0][8]  = 0
        b_matrix[0][9]  = -3*(x+x*y)/(4*a**2)
        b_matrix[0][10] = -((3*a*x+3*a*x*y-a-a*y)/4)/a**2
        b_matrix[0][11] = 0
        b_matrix[1][0]  = -3*(y-x*y)/(4*b**2)
        b_matrix[1][1]  = 0
        b_matrix[1][2]  = -((3*b*y-3*b*x*y-b+b*x)/4)/b**2
        b_matrix[1][3]  = -3*(y+x*y)/(4*b**2)
        b_matrix[1][4]  = 0
        b_matrix[1][5]  = -((3*b*y+3*b*x*y-b-b*x)/4)/b**2
        b_matrix[1][6]  = -3*(-y-x*y)/(4*b**2)
        b_matrix[1][7]  = 0
        b_matrix[1][8]  = -((3*b*y+3*b*x*y+b+b*x)/4)/b**2
        b_matrix[1][9]  = -3*(-y+x*y)/(4*b**2)
        b_matrix[1][10] = 0
        b_matrix[1][11] = -((3*b*y-3*b*x*y+b-b*x)/4)/b**2
        b_matrix[2][0]  = -2*(1/2-3*x**2/8-3*y**2/8)/(a*b)
        b_matrix[2][1]  = -2*(-3/8*a*x**2+a*x/4+a/8)/(a*b)
        b_matrix[2][2]  = -2*(-3/8*b*y**2+b*y/4+b/8)/(a*b)
        b_matrix[2][3]  = -2*(-1/2+3*x**2/8+3*y**2/8)/(a*b)
        b_matrix[2][4]  = -2*(-3/8*a*x**2-a*x/4+a/8)/(a*b)
        b_matrix[2][5]  = -2*(3/8*b*y**2-b*y/4-b/8)/(a*b)
        b_matrix[2][6]  = -2*(1/2-3*x**2/8-3*y**2/8)/(a*b)
        b_matrix[2][7]  = -2*(3/8*a*x**2+a*x/4-a/8)/(a*b)
        b_matrix[2][8]  = -2*(3/8*b*y**2+b*y/4-b/8)/(a*b)
        b_matrix[2][9]  = -2*(-1/2+3*x**2/8+3*y**2/8)/(a*b)
        b_matrix[2][10] = -2*(3/8*a*x**2-a*x/4-a/8)/(a*b)
        b_matrix[2][11] = -2*(-3/8*b*y**2-b*y/4+b/8)/(a*b)
        return b_matrix

    def get_load_vector(self, q):
        a = self.a
        b = self.b
        q = 4*q*a*b * np.array([1/4, a/12, b/12, 1/4, -a/12, b/12, 1/4, -a/12, -b/12, 1/4, a/12, -b/12])
        self.e_loads.append(q)

    def get_internal_forces(self, c_n, r_t):
        a = []
        for i in range(12):
            if (self.vec[i] in c_n):
                indx = list(c_n).index(self.vec[i])
                a.append(r_t[indx])
            else:
                a.append(0)
        a = np.array(a)

        d_mtrx =    self.E*self.h**3/(12*(1-self.mi**2)) * np.array([
                    [1,       self.mi, 0             ],
                    [self.mi, 1,       0             ],
                    [0,       0,       (1-self.mi)/2 ] ] )

        stress = [ 0, 0, 0 ]
        for i in range(4):
            stress += d_mtrx @ self.b_mat[i] @ a

        self.moments = stress/4

# Definition of construction

LX = 6
LY = 9

mesh = .75

nx = int(LX/mesh)
ny = int(LY/mesh)


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
        l_elems.append( Element(l_nodes[n0], l_nodes[n1], l_nodes[n2], l_nodes[n3], h = 0.200, E = 32.8*1e9) )

## LOAD CASES

# for i in l_elems:
#     i.get_load_vector(2000)

for i in l_elems:
    i.get_load_vector(1000)

# Assembly of Global stiffness B_matrix
n_e       = len(l_elems)
n_n       = len(l_nodes)
dofs      = n_n*3
code_nums = list(range(dofs))

load = [0 for i in range(dofs)]
for e in l_elems:
    for l in e.e_loads:
        for i in range(12):             # because element matrix is 12x12 shape
            load[e.vec[i]] += l[i]

# # 4 points around
# boundary = []
# for i in l_nodes:
#     if i.co_x == 0 and i.co_y == LY:
#         boundary.append(i.w)
#     if i.co_x == LX and i.co_y == 0:
#         boundary.append(i.w)
#     if i.co_x == LX and i.co_y == LY:
#         boundary.append(i.w)
#     if i.co_x == 0 and i.co_y == 0:
#         boundary.append(i.w)
# deleto = boundary

# # 4 Edges around
# boundary = []
# for i in l_nodes:
#     if i.co_x == 0:
#         boundary.append(i.w)
#     if i.co_x == LX:
#         boundary.append(i.w)
#     if i.co_y == 0:
#         boundary.append(i.w)
#     if i.co_y == LY:
#         boundary.append(i.w)
# deleto = boundary

# 4 points around
boundary = []
for i in l_nodes:
    # if i.co_x == 0 and i.co_y == LY:
    #     boundary.append(i.w)
    if i.co_x == 0 and i.co_y >= 0.5*LY:
        boundary.append(i.w)
    if i.co_x == LX and i.co_y == 0:
        boundary.append(i.w)
    if i.co_x == LX and i.co_y == LY:
        boundary.append(i.w)
    if i.co_x == 0 and i.co_y == 0:
        boundary.append(i.w)
    if i.co_y == LY/2 and i.co_x <= LX/2:
        boundary.append(i.w)

    # if i.co_x == 0.5 and i.co_y == 0.5:
    #     boundary.append(i.w)
    # if i.co_x == 0 and i.co_y <= 1:
    #     boundary.append(i.w)
    #     boundary.append(i.fi_x)
    #     boundary.append(i.fi_y)
    # if i.co_x == 0.5 and i.co_y <= 1:
    #     boundary.append(i.w)
    #     boundary.append(i.fi_x)
    #     boundary.append(i.fi_y)
    # if i.co_x == 1 and i.co_y <= 1:
    #     boundary.append(i.w)
    #     boundary.append(i.fi_x)
    #     boundary.append(i.fi_y)
deleto = boundary

# # 4 Edges around
# boundary = []
# for i in l_nodes:
#     if i.co_x == 0:
#         boundary.append(i.w)
#     if i.co_x == LX:
#         boundary.append(i.w)
#     if i.co_y == 0:
#         boundary.append(i.w)
#     if i.co_y == LY:
#         boundary.append(i.w)
# deleto = boundary


K_gl = np.zeros( (dofs, dofs) )
for i in l_elems:
    for j in range(12):                 # because element matrix is 12x12 shape
        for k in range(12):             # because element matrix is 12x12 shape
            K_gl[i.vec[j],i.vec[k]] += i.k_el[j,k]

# np.savetxt("2elems_full.csv",K_gl, fmt = "%.2e")

K_gl = np.delete(K_gl, deleto, axis = 0)
K_gl = np.delete(K_gl, deleto, axis = 1)
load = np.delete(load, deleto, axis = 0)

code_nums = np.delete(code_nums, deleto, axis = 0)

delta = np.linalg.inv(K_gl)
r_tot = np.matmul(delta, load)

## APPLICATION OF DEFORMATIONS
for i in range(len(r_tot)):
    if code_nums[i]%3 == 0:
        l_nodes[code_nums[i]//3].apply_deformation(r_tot[i])

## GET INTERNAL FORCES ON ELEMENTS
for i in l_elems:
    i.get_internal_forces(code_nums, r_tot)

print(l_elems[0].moments)

x = np.arange(0+0.5*mesh, LX, mesh)
y = np.arange(0+0.5*mesh, LY, mesh)
X, Y = np.meshgrid(x, y)
Z = np.zeros(np.shape(X))

cunt = 0
for i in range(len(Z)):
    for j in range(len(Z[0])):
        Z[i,j] = int(l_elems[cunt].moments[0])
        cunt +=1
print(Z)

red_r_tot  = []
red_c_nums = []
for i in range(len(code_nums)):
    if code_nums[i]%3 == 0:
        red_r_tot.append(r_tot[i])
        red_c_nums.append(code_nums[i])

r_max = max(red_r_tot)
r_min = min(red_r_tot)
c_n_max = red_c_nums[red_r_tot.index(r_max)]
c_n_min = red_c_nums[red_r_tot.index(r_min)]

print(r_max, r_min)

_3D = 0
if _3D:
    scale = 2
    # PLOT
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d as plt3d
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    red_r_tot  = []
    red_c_nums = []
    for i in range(len(code_nums)):
        if code_nums[i]%3 == 0:
            red_r_tot.append(r_tot[i])
            red_c_nums.append(code_nums[i])

    r_max = max(red_r_tot)
    r_min = min(red_r_tot)
    c_n_max = red_c_nums[red_r_tot.index(r_max)]
    c_n_min = red_c_nums[red_r_tot.index(r_min)]

    if LX > LY:
        ax.set_xlim( (0, LX) )
        ax.set_ylim( (-abs(LX-LY)/2, LY+(abs(LX-LY)/2)) )
        ax.set_zlim( (min(-scale*r_max,0), max(-scale*r_min,0)) )
    elif LX < LY:
        ax.set_xlim( (-abs(LX-LY)/2, LX+(abs(LX-LY)/2)) )
        ax.set_ylim( (0,LY) )
        ax.set_zlim( (min(-scale*r_max,0), max(-scale*r_min,0)) )
    else:
        ax.set_xlim( (0,LX) )
        ax.set_ylim( (0,LY) )
        ax.set_zlim( (min(-scale*r_max,0), max(-scale*r_min,0)) )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Remove gray panes and axis grid
    ax.xaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.fill = False
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.fill = False
    ax.zaxis.pane.set_edgecolor('white')
    ax.grid(False)
    # Remove z-axis
    ax.w_zaxis.line.set_lw(0.)
    ax.set_zticks([])

    X = np.zeros( (ny+1, nx+1) )
    Y = np.zeros( (ny+1, nx+1) )
    Z = np.zeros( (ny+1, nx+1) )
    Z_= np.zeros( (ny+1, nx+1) )

    for i in range(ny+1):
        for j in range(nx+1):
            n = Nodes_[i,j]
            X[i,j] = l_nodes[n].co_x
            Y[i,j] = l_nodes[n].co_y
            Z[i,j] = 0
            Z_[i,j]= l_nodes[n].def_z

    surf_undef = ax.plot_surface(X, Y, Z,
                 linewidth=1, antialiased=True, alpha = 0.1 )

    surf_def   = ax.plot_surface(X, Y, Z_, cmap=cm.coolwarm,
                 linewidth=0, antialiased=True, alpha = 0.5 )

    for i in deleto:
        ax.plot([l_nodes[i//3].co_x], [l_nodes[i//3].co_y], [0.], markerfacecolor='k', markeredgecolor='k', marker='o', markersize=1, alpha=1.)

    ax.plot([l_nodes[c_n_max//3].co_x], [l_nodes[c_n_max//3].co_y], [l_nodes[c_n_max//3].def_z], markeredgecolor='b', marker='x', markersize=7, alpha=0.7)
    ax.plot([l_nodes[c_n_min//3].co_x], [l_nodes[c_n_min//3].co_y], [l_nodes[c_n_min//3].def_z], markeredgecolor='r', marker='x', markersize=7, alpha=0.7)

    plt.show()


_2D = 0
if _2D:
    import matplotlib.pyplot as plt

    X = [0, LX, LX, 0 , 0]
    Y = [0, 0,  LY, LY, 0]
    fig = plt.figure(figsize=(LX, LY))
    plt.plot(X, Y, linewidth=0.5)

    for i in deleto:
        plt.plot([l_nodes[i//3].co_x], [l_nodes[i//3].co_y], markerfacecolor='k', markeredgecolor='k', marker='o', markersize=1, alpha=1.)

    x = np.arange(0+0.5*mesh, LX, mesh)
    y = np.arange(0+0.5*mesh, LY, mesh)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(np.shape(X))

    cunt = 0
    for i in range(len(Z)):
        for j in range(len(Z[0])):
            Z[i,j] = l_elems[cunt].moments[2]
            cunt +=1

    plt.contour(X, Y, Z)
    # ax.clabel(CS, inline=True, fontsize=5)

    plt.title('A Basic Line Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    plt.show()
