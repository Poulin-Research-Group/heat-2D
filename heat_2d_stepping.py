#!/usr/bin/env python
# heat_2d_stepping.py
#
# This will solve the 2-D heat equation using the stepping equations
from __future__ import division
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

np.set_printoptions(threshold=np.inf)  # make sure ENTIRE array is printed

# CPU warmup
np.random.rand(500, 500).dot(np.random.rand(500, 500))

# read from STDIN
if len(sys.argv) > 1:
    Mx = 2**(int(sys.argv[1]))
    My = 2**(int(sys.argv[2]))
    # N = 10**(int(sys.argv[3]))
    i = int(sys.argv[4])
    writeToFile = bool(int(sys.argv[5]))
else:
    Mx = 256     # total x points (inner)
    My = 256     # total y points (inner)
    # N = 10000   # time steps
    i = None
    writeToFile = False

# total number of points
M = (Mx+2) * (My+2)

# x conditions
x0 = 0                          # start
xf = 1                          # end
dx = (xf-x0)/(Mx+1)             # spatial step size
x  = np.linspace(x0, xf, Mx+2)  # points

# y conditions
y0 = 0
yf = 1
dy = (yf-y0)/(My+1)
y  = np.linspace(y0, yf, My+2)

# temporal conditions
N  = 1000         # time steps
t0 = 0            # start
tf = 300          # end
dt = (tf - t0)/N  # time step size
t  = np.linspace(t0, tf, N)

# coefficients
k = 0.0002
Kx = 0.02                # PDE coeff for x terms
Ky = 0.01
C  = 1 - 2*(Kx + Ky)


# initial condition function
def f(x, y):
    # x, y can be arrays
    return np.sin(np.pi*x) * np.sin(np.pi*y)

# initial solution
u  = np.array([f(x, i) for i in y])             # not ideal, but good for now
un = np.empty((My+2, Mx+2), dtype=np.float64)   # My+2 columns of Mx+2 entries

# initialize the final solution vector (u0, ..., u_m+1)
# U = np.empty((My+2, Mx+2, N), dtype=np.float64)
# U[:, :, 0] = u

t_start = time.time()

# plt.ion()

# Loop over time
for j in range(1, N):
    un[1:My+1, 1:Mx+1] = C*u[1:My+1:, 1:Mx+1] + \
                         Kx*(u[1:My+1:, 0:Mx] + u[1:My+1, 2:Mx+2]) + \
                         Ky*(u[0:My,  1:Mx+1] + u[2:My+2, 1:Mx+1])

    # Force Boundary Conditions
    un[0, :]    = 0.0  # first row
    un[My+1, :] = 0.0  # last row
    un[:, 0]    = 0.0  # first col
    un[:, Mx+1] = 0.0  # last col

    # save solution
    # U[:,:,j] = un

    # Update solution
    u = un

t_final = time.time()
print t_final - t_start


if writeToFile:
    # write time to a file
    F = open('./tests/ser-step/M%s.txt' % (sys.argv[1].zfill(2)), 'r+')
    F.read()
    F.write('%f\n' % (t_final - t_start))
    F.close()

if i == 0:
    G = open('./tests/ser-step/solution-M%s.txt' % (sys.argv[1].zfill(2)), 'r+')
    G.read()
    G.write('%s\n' % str(u))
    G.close()


sys.exit()

# PLOTTING
fig = plt.figure()
ims = []
for j in xrange(N):
    print j
    ims.append((plt.pcolormesh(x, y, U[:, :, j], norm=plt.Normalize(0, 1)), ))  # THIS IS A TUPLE

im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000, blit=False)

im_ani.save('im.mp4')
plt.show()

sys.exit()
