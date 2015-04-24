from __future__ import division
import numpy as np
import sys
from setup import solver, Params, METHODS, UPDATERS
from mpi4py import MPI


# map method name to the actual function
get_updater = dict(zip(METHODS, UPDATERS))


# initial condition function
def f(x, y):
    # x, y can be arrays
    return np.sin(np.pi*x) * np.sin(np.pi*y)


# boundary condition (BC) functions; four functions imposing BCs must be
# defined: one for serial solutions, and then one each for solutions
# parallelized in only x, only y, and both x and y.
# TODO ==============
# add some BC functions...

# handling command line arguments, e.g.
#
#   python stepping_mpi_2D.py numpy 1 1 2 2
#
# will run this script using numpy to calculate the next solution, px = 1,
# py = 1, sc_x = 2, sc_y = 2

if len(sys.argv) > 1:
    argv = sys.argv[1:]
    method = argv[0]
    px   = int(argv[1])
    py   = int(argv[2])
    sc_x = int(argv[3])
    sc_y = int(argv[4])

# if there are no command line arguments...
else:
    # number of processors to use in each direction
    px = 1
    py = 1

    # scaling parameters
    sc_x = 2
    sc_y = 2

    # method to use; options are 'numpy', 'f2py77', 'f2py90'
    method = 'numpy'


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

# Define type of BCs ('P' = periodic)
BC_type = 'P'
BC_s  = None
BC_x  = None
BC_y  = None
BC_xy = None

# if SAVE_TIME is True, then the total time to solve the problem will be saved
# to a file named filename_time
SAVE_TIME = True
filename_time = './tests/%s/%dscx_%dscy_%dpx_%dpy.txt' % (method, sc_x, sc_y, px, py)

# if ANIMATE is True, then an animation of the solution will be saved to a file
# named filename_anim
ANIMATE = False
filename_anim = './anims/anim_%dpx_%dpy.mp4' % (px, py)

# if SAVE_SOLN is True, then the solution at every time step will be saved to a
# file named filename_soln
SAVE_SOLN = False
filename_soln = './solns/soln_%dpx_%dpy.txt' % (px, py)

# DO NOT ALTER =======================================
Updater = get_updater[method]

params = Params()
params.set_x_vars([x0, xf, dx, Nx, nx])
params.set_y_vars([y0, yf, dy, Ny, ny])
params.set_t_vars([t0, tf, dt, Nt])
params.set_consts([C, Kx, Ky])
params.set_bc_funcs([BC_s, BC_x, BC_y, BC_xy])
params.ics = f
params.bcs_type = BC_type
params.filename_time = filename_time
params.filename_anim = filename_anim
params.filename_soln = filename_soln

solver(Updater, params, px, py, SAVE_TIME, ANIMATE, SAVE_SOLN)
