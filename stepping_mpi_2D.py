from __future__ import division
import numpy as np
import sys
from setup import solver, comm, calc_u, heatf, heatf90, METHODS, UPDATERS
from fjp_helpers.animator import mesh_animator
from fjp_helpers.misc import writer


# initial condition function
def f(x, y):
    # x, y can be arrays
    return np.sin(np.pi*x) * np.sin(np.pi*y)


# number of processors to use in each direction
px = 2
py = 2

# scaling parameters
sc_x = 1
sc_y = 1

# number of spatial points
Nx = 128*sc_x
Ny = 128*sc_y

# number of (x, y) spatial points per processes dedicated to respective directions
nx = Nx/px
ny = Ny/py

# x conditions
x0 = 0                  # start
xf = 1                  # end
dx = (xf-x0)/(Nx+1)     # spatial step size

# y conditions
y0 = 0
yf = 1
dy = (yf-y0)/(Ny+1)

# temporal conditions
Nt = 1000          # number of time steps
t0 = 0
tf = 300
dt = (tf - t0)/Nt

# PDE coefficients
Kx = 0.02
Ky = 0.01
C  = 1 - 2*(Kx + Ky)

# Method to use (numpy, f2py-f77, f2py-f90 ...)
Updater = calc_u

# DO NOT ALTER =======================================
x_vars = [x0, xf, dx, Nx, nx]
y_vars = [y0, yf, dy, Ny, ny]
t_vars = [t0, tf, dt, Nt]
consts = [C, Kx, Ky]


"""
# handling command line arguments
if len(sys.argv) > 1:
    get_updater = dict(zip(METHODS, UPDATERS))

    argv = sys.argv[1:]
    updater = get_updater[argv[0]]
    sc = int(argv[1])
    px = int(argv[2])
    py = int(argv[3])

    (Updater, sc, px, py)
    comm.Barrier()
    sys.exit()
"""

solver(Updater, x_vars, y_vars, t_vars, consts, f, sc_x, sc_y, px, py)
