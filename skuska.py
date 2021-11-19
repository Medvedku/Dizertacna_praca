import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

np.set_printoptions(precision=1)

delta = .25
x = np.arange(-3.0, 3.0, delta)
print(x,len(x),"x")
y = np.arange(-2.0, 2.0, delta)
print(y,len(y),"y")
X, Y = np.meshgrid(x, y)
print(X, "X")
print(Y, "Y")
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2
print(Z)

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
ax.clabel(CS, inline=True, fontsize=5)
ax.set_title('Sexi kontúry')
plt.savefig("plotino.svg", format="svg")
plt.show()
