#!/usr/bin/env python
# heat_2d_stepping_mpi.py
#
# This will solve the 2-D heat equation in parallel using mpi4py

from __future__ import division
import sys
from mpi4py import MPI
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
comm = MPI.COMM_WORLD

np.set_printoptions(threshold=np.inf)  # make sure ENTIRE array is printed

rank = comm.Get_rank()   # this process' ID
p = comm.Get_size()    # number of processors


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
Kx = np.float64(0.02)                # PDE coeff for x terms
Ky = np.float64(0.01)
C  = 1 - 2*(Kx + Ky)


# initial condition function
def f(x, y):
    # x, y can be arrays
    return np.sin(np.pi*x) * np.sin(np.pi*y)

# BUILD ZE GRID
u   = np.array([f(x, j) for j in y])     # process' slice of soln
un  = np.empty((My+2, mx+2), dtype='d')  # process' slice of NEW soln
col = np.empty(My+2, dtype='d')

"""
if rank == 0:
    xg = np.linspace(x0, xf, Mx+2)
    ug = np.array([f(xg, j) for j in y])[1:-1, 1:-1].flatten()
    U  = np.empty((My, Mx, N), dtype=np.float64)
    U[:, :, 0] = ug.reshape(My, Mx)
    t  = np.linspace(t0, tf, N)
else:
    ug = None
"""

tags = dict([(j, j+5) for j in xrange(p)])


def writer(t_final, u, writeToFile, i, subdir, method):
    if writeToFile:
        # write time to a file
        F = open('./tests/%s/%s/p%d-M%s.txt' % (subdir, method, p, str(M).zfill(2)), 'r+')
        F.read()
        F.write('%f\n' % t_final)
        F.close()

    # write the solution to a file, but only once!
    if i == 0:
        G = open('./tests/par-step/solution-p%d.txt' % p, 'r+')
        G.read()
        G.write('%s\n' % str(u))
        G.close()
