import numpy as np
import timeit
st1 = timeit.default_timer()

pi = np.pi
np.set_printoptions(precision=3)

spiel = True



def check_symmetric(a, rtol=1e-05, atol=1e-05):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def Brick_K(l_x = 1, l_y = 1, l_z = 1, nu = 0.3, E = 1):
    C_m = np.array([ [ 1-nu, nu,   nu,   0,          0,          0          ],
                     [ nu,   1-nu, nu,   0,          0,          0          ],
                     [ nu,   nu,   1-nu, 0,          0,          0          ],
                     [ 0,    0,    0,    (1-2*nu)/2, 0,          0          ],
                     [ 0,    0,    0,    0,          (1-2*nu)/2, 0          ],
                     [ 0,    0,    0,    0,          0,          (1-2*nu)/2]]) * E/((1+nu)*(1-2*nu))

    GaussPoint = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])

    coordinates = np.array([    [-l_x/2, -l_y/2, -l_z/2],
                                [-l_x/2,  l_y/2, -l_z/2],
                                [ l_x/2,  l_y/2, -l_z/2],
                                [ l_x/2, -l_y/2, -l_z/2],
                                [-l_x/2, -l_y/2,  l_z/2],
                                [-l_x/2,  l_y/2,  l_z/2],
                                [ l_x/2,  l_y/2,  l_z/2],
                                [ l_x/2, -l_y/2,  l_z/2], ])

    K = np.zeros( (24, 24) )    # prázdna matica
    for xi1 in GaussPoint:
        for xi2 in GaussPoint:
            for xi3 in GaussPoint:
                d_shape = 1/8 * np.array([
                [ -(1-xi2)*(1-xi3),  (1-xi2)*(1-xi3),  (1+xi2)*(1-xi3), -(1+xi2)*(1-xi3),
                  -(1-xi2)*(1+xi3),  (1-xi2)*(1+xi3),  (1+xi2)*(1+xi3), -(1+xi2)*(1+xi3)],
                [ -(1-xi1)*(1-xi3), -(1+xi1)*(1-xi3),  (1+xi1)*(1-xi3),  (1-xi1)*(1-xi3),
                  -(1-xi1)*(1+xi3), -(1+xi1)*(1+xi3),  (1+xi1)*(1+xi3),  (1-xi1)*(1+xi3)],
                [ -(1-xi1)*(1-xi2), -(1+xi1)*(1-xi2), -(1+xi1)*(1+xi2), -(1-xi1)*(1+xi2),
                   (1-xi1)*(1-xi2),  (1+xi1)*(1-xi2),  (1+xi1)*(1+xi2),  (1-xi1)*(1+xi2)] ])

                Jacobian_Matrix = np.matmul(d_shape, coordinates)
                    # Jacobiho matica (matica parciálnych derivácií)
                Help_matrix = np.matmul(np.linalg.inv(Jacobian_Matrix),d_shape)
                    # Výpočet pomocnej matice potrebnej na zostrojenie B - operátora
                B = np.zeros( (6, 24) )
                    # Výpočet B - operátora
                for i in range(3):
                    for j in range(8):
                        B[i][3*j+(i)] = Help_matrix[i][j]
                for k in range(8):
                    B[3][3*k+0] = Help_matrix[1][k] # 4. riadok
                    B[3][3*k+1] = Help_matrix[0][k] # 4. riadok

                    B[4][3*k+2] = Help_matrix[1][k] # 5. riadok
                    B[4][3*k+1] = Help_matrix[2][k] # 5. riadok

                    B[5][3*k+0] = Help_matrix[2][k] # 6. riadok
                    B[5][3*k+2] = Help_matrix[0][k] # 6. riadok

                K = K + np.matmul( np.matmul( np.transpose(B),C_m ),B ) * np.linalg.det(Jacobian_Matrix)
    return K

class Element:
    num_of_elem = 0
    def __init__(self, ndarray: np.ndarray):
        self.el_id = Element.num_of_elem
        self.nda = ndarray
        self.nodes = [ int(self.nda[0][0][0]), int(self.nda[0][1][0]), int(self.nda[0][1][1]), int(self.nda[0][0][1]), int(self.nda[1][0][0]), int(self.nda[1][1][0]), int(self.nda[1][1][1]), int(self.nda[1][0][1]) ]

        Element.num_of_elem += 1

class Node:
    num_of_nodes = 0
    def __init__(self, xyz: list):
        self.nd_id = Node.num_of_nodes
        self.u = Node.num_of_nodes*3 + 0
        self.v = Node.num_of_nodes*3 + 1
        self.w = Node.num_of_nodes*3 + 2
        self.coor = xyz
        self.coor_abs = [self.coor[0] * e_lx, self.coor[1] * e_ly, self.coor[2] * e_lz]
        self.coor_def = [self.coor[0] * e_lx, self.coor[1] * e_ly, self.coor[2] * e_lz]

        Node.num_of_nodes += 1

    def app_deformation(self, defs_: np.ndarray, indi_: np.ndarray):
        for i in range(len(indi_)):
            if indi_[i] == self.u:
                self.coor_def[0] += defs_[i]
            if indi_[i] == self.v:
                self.coor_def[1] += defs_[i]
            if indi_[i] == self.w:
                self.coor_def[2] += defs_[i]

# Dimensions of cube
LX, LY, LZ = 0.015, 0.015, 0.015

#Number of nodes in x, y, z direction, (not elements)!!
vx, vy, vz = 6,6,6       # → - x, ↙ - y, ↥ - z

#Dimensions of elements
e_lx, e_ly, e_lz = LX/(vx-1), LY/(vy-1), LZ/(vz-1)

Nodes = np.ndarray(shape=(vz, vy, vx, 1))
defs = np.ndarray(shape=(vz, vy, vx, 3))

Elements = []
nodes_   = []

cunt = 0
for z in range(vz):
    for y in range(vy):
        for x in range(vx):
            Nodes[z][y][x] = cunt
            nodes_.append(Node([x, y, z]))
            cunt += 1

for z in range(vz-1):
    for y in range(vy-1):
        for x in range(vx-1):
            Elements.append(Element(Nodes[z:z+2, y:y+2, x:x+2]))

K_el = Brick_K(l_x = e_lx, l_y = e_ly, l_z = e_lz, nu = 0.25, E=1e10)

if spiel == True:
    print("Is stiffness matrix of an element symetric?: {0}".format(check_symmetric(K_el)))

G_K = np.zeros((cunt*3,cunt*3))


deleto = []

for i in nodes_:
    if i.coor[0] == 0:
        deleto.append(i.u)
    if i.coor[1] == 0:
        deleto.append(i.v)
    if i.coor[2] == 0:
        deleto.append(i.w)
print(deleto)

# for i in nodes_:
#     if i.coor[2] == vx-1:
#         if deleto.count(i.u) == 0:
#             deleto.append(i.u)
#         if deleto.count(i.v) == 0:
#             deleto.append(i.v)
#         if deleto.count(i.w) == 0:
#             deleto.append(i.w)
# print(deleto)

deformations_iden = list(range(len(nodes_)*3))
F = np.zeros((1,len(nodes_)*3), dtype = np.int)
F = list(F[0])
F[-1] = -1000

for i in Elements:
    k = []
    for j in i.nodes:
        k.append(j*3)
        k.append(j*3+1)
        k.append(j*3+2)
        for m in range(len(k)):
            for n in range(len(k)):
                G_K[k[m]][k[n]] += K_el[m][n]

if spiel == True:
    print("Is global stiffness matrix symetric?: {0}".format(check_symmetric(G_K)))
    print("Global stiffnes matrix assembled {0}s after start.".format(round(timeit.default_timer() - st1, 5)))

G_K = np.delete(G_K, deleto, axis = 0)
G_K = np.delete(G_K, deleto, axis = 1)
F = np.delete(F, deleto, axis = 0)
deformations_iden = np.delete(deformations_iden, deleto, axis = 0)

if spiel == True:
    print("Is global stiffness matrix of an element symetric after reduction?: {0}".format(check_symmetric(G_K)))
    print("Boundary condtions applied {0}s after start.".format(round(timeit.default_timer() - st1, 5)))

defor = np.matmul(np.linalg.inv(G_K), F)

if spiel == True:
    print("System of equations solved {0}s after start.".format(round(timeit.default_timer() - st1, 5)))

def_scale = 35
scaled_defs = def_scale * defor


for i in nodes_:
    i.app_deformation(scaled_defs, deformations_iden)


if spiel == True:
    # PLOT
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d as plt3d
    fig = plt.figure()
    # fig.set_size_inches(vx*10,vy*10)
    ax = fig.add_subplot(111, projection='3d')
    range_setup = 0.05
    ax.set_xlim( (-range_setup*LX, LX*(1+range_setup)) )
    ax.set_ylim( (-range_setup*LY, LY*(1+range_setup)) )
    ax.set_zlim( (-range_setup*LZ, LZ*(1+range_setup)) )

    lines_und = [ [], [], [], [], [], [], [], [], [], [], [], [] ]
    for i in nodes_:
        if i.coor[0] == 0 and i.coor[2] == 0:
            lines_und[0].append(i.nd_id)
        if i.coor[1] == 0 and i.coor[2] == 0:
            lines_und[3].append(i.nd_id)
        if i.coor[0] == vx-1 and i.coor[2] == 0:
            lines_und[2].append(i.nd_id)
        if i.coor[1] == vy-1 and i.coor[2] == 0:
            lines_und[1].append(i.nd_id)
        if i.coor[0] == 0 and i.coor[2] == vz-1:
            lines_und[8].append(i.nd_id)
        if i.coor[1] == 0 and i.coor[2] == vz-1:
            lines_und[11].append(i.nd_id)
        if i.coor[0] == vx-1 and i.coor[2] == vz-1:
            lines_und[10].append(i.nd_id)
        if i.coor[1] == vy-1 and i.coor[2] == vz-1:
            lines_und[9].append(i.nd_id)
        if i.coor[0] == 0 and i.coor[1] == 0:
            lines_und[4].append(i.nd_id)
        if i.coor[0] == vx-1 and i.coor[1] == 0:
            lines_und[7].append(i.nd_id)
        if i.coor[0] == vx-1 and i.coor[1] == vy-1:
            lines_und[6].append(i.nd_id)
        if i.coor[0] == 0 and i.coor[1] == vy-1:
            lines_und[5].append(i.nd_id)

    x_s = []
    y_s = []
    z_s = []

    for k in lines_und:
        for j in range(len(k)-1):
            x_s.append( (nodes_[k[j]].coor_abs[0], nodes_[k[j+1]].coor_abs[0]) )
            y_s.append( (nodes_[k[j]].coor_abs[1], nodes_[k[j+1]].coor_abs[1]) )
            z_s.append( (nodes_[k[j]].coor_abs[2], nodes_[k[j+1]].coor_abs[2]) )

    for i in range(len(x_s)):
        xs_ = x_s[i]
        ys_ = y_s[i]
        zs_ = z_s[i]
        line = plt3d.art3d.Line3D(xs_, ys_, zs_, c="b", lw="0.25")
        ax.add_line(line)

    x_s = []
    y_s = []
    z_s = []

    for k in lines_und:
        for j in range(len(k)-1):
            x_s.append( (nodes_[k[j]].coor_def[0], nodes_[k[j+1]].coor_def[0]) )
            y_s.append( (nodes_[k[j]].coor_def[1], nodes_[k[j+1]].coor_def[1]) )
            z_s.append( (nodes_[k[j]].coor_def[2], nodes_[k[j+1]].coor_def[2]) )

    for i in range(len(x_s)):
        xs_ = x_s[i]
        ys_ = y_s[i]
        zs_ = z_s[i]
        line = plt3d.art3d.Line3D(xs_, ys_, zs_, c="r", lw="0.25")
        ax.add_line(line)

    print("Showing plot {0}s after start.".format(round(timeit.default_timer() - st1, 5)))

    plt.show()

quit()
