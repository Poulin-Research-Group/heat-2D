#!/usr/bin/env python
# heat_2d_stepping_mpi.py
#
# This will solve the 2-D heat equation in parallel using mpi4py

from __future__ import division
import sys
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
comm = MPI.COMM_WORLD

np.set_printoptions(threshold=np.inf)  # make sure ENTIRE array is printed

rank = comm.Get_rank()   # this process' ID
p = comm.Get_size()    # number of processors

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

mx = Mx/p   # x-grid points per process

# total number of points
M = (Mx+2) * (My+2)

# x conditions
x0 = 0                       # start
xf = 1                       # end
dx = (xf-x0)/(Mx+1)          # spatial step size
# this takes the interval [x0,xf] and splits it equally among all processes
x = np.linspace(x0 + rank*(xf-x0)/p, x0 + (rank+1)*(xf-x0)/p, mx+2)

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
k  = 0.0002
Kx = 0.02                # PDE coeff for x terms
Ky = 0.01
C  = 1 - 2*(Kx + Ky)


# initial condition function
def f(x, y):
    # x, y can be arrays
    return np.sin(np.pi*x) * np.sin(np.pi*y)

# BUILD ZE GRID
u  = np.array([f(x, i) for i in y])     # process' slice of soln
un = np.empty((My+2, mx+2), dtype='d')  # process' slice of NEW soln

"""
if rank == 0:
    xg= np.linspace(x0, xf, Mx+2)
    ug= np.array([f(xg,i) for i in y])[1:-1, 1:-1].flatten()
    U = np.empty((My,Mx,N), dtype=np.float64)
    U[:, :, 0] = ug.reshape(My,Mx)
    t = np.linspace(t0, tf, N)
else:
    ug = None
"""

comm.Barrier()         # start MPI timer
t_start = MPI.Wtime()

# loop through time
for j in range(1, N):

    # Send u[:, 1] to ID-1
    if 0 < rank:
        comm.send(u[:, 1], dest=rank-1, tag=1)

    # Receive u[mx+1] to ID+1
    if rank < p-1:
        u[:, mx+1] = comm.recv(source=rank+1, tag=1)

    # Send u[mx] to ID+1
    if rank < p-1:
        comm.send(u[:, mx], dest=rank+1, tag=2)

    # Receive u[0] to ID-1
    if 0 < rank:
        u[:, 0] = comm.recv(source=rank-1, tag=2)

    un[1:My+1, 1:mx+1] = C*u[1:My+1:, 1:mx+1] + \
                         Kx*(u[1:My+1:, 0:mx] + u[1:My+1, 2:mx+2]) + \
                         Ky*(u[0:My,  1:mx+1] + u[2:My+2, 1:mx+1])

    # Force Boundary Conditions
    if rank == 0:
        un[:, 0]    = 0.0  # first col
    elif rank == p-1:
        un[:, mx+1] = 0.0  # last col

    un[0, :]    = 0.0  # first row
    un[My+1, :] = 0.0  # last row

    # update soln
    u = un

    """
    # Gather parallel vectors to a serial vector
    comm.Gather(u[1:My+1, 1:mx+1].flatten(), ug, root=0)
    if rank == 0:
        # evenly split ug into a list of p parts
        temp = np.array_split(ug, p)
        # reshape each part
        temp = [a.reshape(My,mx) for a in temp]
        U[:, :, j] = np.hstack(temp)
    """


comm.Barrier()
t_final = (MPI.Wtime() - t_start)  # stop MPI timer


if rank == 0:
    if writeToFile:
        # write time to a file
        F = open('./tests/par-step/p%d-M%s.txt' % (p, sys.argv[1].zfill(2)), 'r+')
        F.read()
        F.write('%f\n' % t_final)
        F.close()

    print t_final

    # write the solution to a file, but only once!
    if i == 0:
        G = open('./tests/par-step/solution-p%d.txt' % p, 'r+')
        G.read()
        G.write('%s\n' % str(u))
        G.close()


sys.exit()

if rank == 0:
    # PLOTTING
    fig = plt.figure()
    ims = []
    for j in xrange(N):
        print j
        ims.append((plt.pcolormesh(xg[1:-1], y[1:-1], U[:, :, j], norm=plt.Normalize(0, 1)), ))

    print 'done creating meshes, attempting to put them together...'
    im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000, blit=False)

    print 'saving...'
    im_ani.save('stepping_mpi.mp4')
    # plt.show()
    print 'saved.'
