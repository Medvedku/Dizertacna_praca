import numpy as np

pi = np.pi

np.set_printoptions(precision=3)

spiel = True

def check_symmetric(a, rtol=1e-05, atol=1e-05):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

l_x = 3
l_y = 5

h = 200 * 1e-3
