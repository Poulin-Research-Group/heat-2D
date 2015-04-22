#!/usr/bin/env python
# heat_2d_setup.py

from __future__ import division
import sys
import numpy as np
import time
from heatFortran import heatf
from heatFortran90 import heatf as heatf90
from mpi4py import MPI
from fjp_helpers.animator import mesh_animator
from fjp_helpers.bc import *
from fjp_helpers.mpi import *
from fjp_helpers.misc import *
comm = MPI.COMM_WORLD


def calc_u(u, C, Kx, Ky):
    u[1:-1, 1:-1] = C*u[1:-1, 1:-1]  + \
                        Kx*(u[1:-1, 0:-2] + u[1:-1, 2:])  + \
                        Ky*(u[0:-2, 1:-1] + u[2:, 1:-1])
    return u


METHODS   = ['numpy', 'f2py77', 'f2py90']
UPDATERS  = [calc_u, heatf, heatf90]


def solver(Updater, x_vars, y_vars, t_vars, consts, f, sc_x, sc_y, px, py):

    p = px * py
    if p != comm.Get_size():
        raise Exception("Incorrect number of cores used; MPI is being run with %d, but %d was inputted." % (comm.Get_size(), p))

    rank  = comm.Get_rank()
    indices = [(i, j) for i in xrange(py) for j in xrange(px)]
    procs = np.arange(p).reshape(py, px)
    locs  = dict(zip(procs.flatten(), indices))   # map rank to location
    loc   = locs[rank]

    left  = np.roll(procs,  1, 1)
    right = np.roll(procs, -1, 1)
    up    = np.roll(procs,  1, 0)
    down  = np.roll(procs, -1, 0)
    rankL = left[loc]
    rankR = right[loc]
    rankU = up[loc]
    rankD = down[loc]

    x0, xf, dx, Nx, nx = x_vars
    y0, yf, dy, Ny, ny = y_vars
    t0, tf, dt, Nt = t_vars
    C, Kx, Ky = consts

    x = create_x(px, rank, x0, xf, dx, nx, Nx)
    y = create_y(px, py, rank, y0, yf, dy, ny, Ny)
    t  = np.linspace(t0, tf, Nt)

    # BUILD ZE GRID
    xx, yy = np.meshgrid(x, y)
    u   = f(xx, yy)
    col = np.empty(ny+2, dtype='d')
    row = np.empty(nx+2, dtype='d')

    tagsL = dict([(j, j+1) for j in xrange(p)])
    tagsR = dict([(j,   p + (j+1)) for j in xrange(p)])
    tagsU = dict([(j, 2*p + (j+1)) for j in xrange(p)])
    tagsD = dict([(j, 3*p + (j+1)) for j in xrange(p)])
    tags  = (tagsL, tagsR, tagsU, tagsD)

    SAVE_GLOBAL_SOLUTION = True
    params = Params([x0, xf, dx, Nx, nx], [y0, yf, dy, Ny, ny], [t0, tf, dt, Nt],
                    [p, px, py], [C, Kx, Ky], [0, 0, 0, 0])

    if px == 1 and py == 1:
        params.set_funcs([f, set_periodic_BC, Updater, None])
        t_total, U = solver_serial(u, params, SAVE_GLOBAL_SOLUTION)

    elif py == 1:
        ranks = (rank,  rankL, rankR)
        tags  = (tagsL, tagsR)
        params.set_funcs([f, set_periodic_BC_y, Updater, send_cols_periodic])
        t_total, U = solver_1D(u, ranks, col, tags, params, SAVE_GLOBAL_SOLUTION)

    elif px == 1:
        ranks = (rank,  rankU, rankD)
        tags  = (tagsU, tagsD)
        params.set_funcs([f, set_periodic_BC_x, Updater, send_rows_periodic])
        t_total, U = solver_mpi_1D(u, ranks, row, tags, params, SAVE_GLOBAL_SOLUTION)

    else:
        ranks = (rank,  rankL, rankR, rankU, rankD)
        tags  = (tagsL, tagsR, tagsU, tagsD)
        params.set_funcs([f, None, Updater, send_periodic])
        t_total, U = solver_mpi_2D(u, ranks, col, row, tags, params, SAVE_GLOBAL_SOLUTION)

    # PLOTTING AND SAVING SOLUTION
    if rank == 0:
        if Updater is calc_u:
            method = 'numpy'
        elif Updater is heatf:
            method = 'f2py77'
        elif Updater is heatf90:
            method = 'f2py90'

        xg = np.linspace(x0 - dx/2, xf + dx/2, Nx+2)
        yg = np.linspace(y0 - dy/2, yf + dy/2, Ny+2)

        mesh_animator(U, xg, yg, nx, ny, Nt, method, p, px, py)

        # writer(t_total, method, sc)
        print t_total

    return t_total


def solver_serial(u, params, save_solution):
    if save_solution:
        t_total, U = solver_serial_helper_g(u, params)
    else:
        t_total, U = solver_serial_helper(u, params)

    return t_total, U


def solver_serial_helper(u, params):
    C,  Kx, Ky = params.consts
    Updater, Set_BCs, = params.updater, params.bcs_func
    Nt = params.Nt

    t_start = time.time()

    # loop through time
    for j in range(1, Nt):
        u = Set_BCs(u)
        u = Updater(u, C, Kx, Ky)

    t_total = (time.time() - t_start)
    return t_total, None


def solver_serial_helper_g(u, params):
    C,  Kx, Ky = params.consts
    Updater, Set_BCs, = params.updater, params.bcs_func
    Nt = params.Nt

    U = create_global_vars(0, params)[1]

    t_start = time.time()

    # loop through time
    for j in range(1, Nt):
        u = Set_BCs(u)
        u = Updater(u, C, Kx, Ky)

        U[:, :, j] = u[1:-1, 1:-1]

    t_total = (time.time() - t_start)
    return t_total, U


def solver_mpi_1D(u, ranks, ghost_arr, tags, params, save_solution):
    """
    Solves the 2D Heat Equation if it is parallelized only in x or y. If
    save_solution is True, then solver_x_helper_g is called. Otherwise,
    solver_x_helper is called.

    Returns
    -------
    tuple (float64, ndarray)   if save_solution is true
    tuple (float64, None)      if save_solution is false
    """

    if save_solution:
        t_total, U = solver_1D_helper_g(u, ranks, ghost_arr, tags, params)
    else:
        t_total, U = solver_1D_helper(u, ranks, ghost_arr, tags, params)

    return t_total, U


def solver_1D_helper(u, ranks, ghost_arr, tags, params):
    p,  px, py = params.p_vars
    C,  Kx, Ky = params.consts
    Updater, Set_BCs, MPI_Func = params.updater, params.bcs_func, params.mpi_func
    Nt = params.Nt

    rank, rankLU, rankRD = ranks
    tagsLU, tagsRD = tags

    comm.Barrier()         # start MPI timer
    t_start = MPI.Wtime()

    # loop through time
    for j in range(1, Nt):
        u = MPI_Func(u, rank, px, ghost_arr, tagsLU, tagsRD, rankLU, rankRD)
        u = Set_BCs(u)
        u = Updater(u, C, Kx, Ky)

    comm.Barrier()
    t_total = (MPI.Wtime() - t_start)  # stop MPI timer
    return t_total, None


def solver_1D_helper_g(u, ranks, ghost_arr, tags, params):
    p,  px, py = params.p_vars
    C,  Kx, Ky = params.consts
    nx, ny, Nt = params.nx, params.ny, params.Nt
    Updater, Set_BCs, MPI_Func = params.updater, params.bcs_func, params.mpi_func

    rank, rankLU, rankRD = ranks
    tagsLU, tagsRD = tags
    ug, U = create_global_vars(rank, params)

    comm.Barrier()         # start MPI timer
    t_start = MPI.Wtime()

    # loop through time
    for j in range(1, Nt):
        u = MPI_Func(u, rank, px, ghost_arr, tagsLU, tagsRD, rankLU, rankRD)
        u = Set_BCs(u)
        u = Updater(u, C, Kx, Ky)

        # Gather parallel vectors to a serial vector
        comm.Gather(u[1:-1, 1:-1].flatten(), ug, root=0)
        if rank == 0:
            U[:, :, j] = get_Uj(u, ug, p, nx, ny)

    comm.Barrier()
    t_total = (MPI.Wtime() - t_start)  # stop MPI timer
    return t_total, U


def solver_mpi_2D(u, ranks, col, row, tags, params, save_solution):
    if save_solution:
        t_total, U = solver_2D_helper_g(u, ranks, col, row, tags, params)
    else:
        t_total, U = solver_2D_helper(u, ranks, col, row, tags, params)

    return t_total, U


def solver_2D_helper(u, ranks, col, row, tags, params):
    p,  px, py = params.p_vars
    C,  Kx, Ky = params.consts
    Updater, MPI_Func = params.updater, params.mpi_func
    Nt = params.Nt

    rank, rankL, rankR, rankU, rankD = ranks
    tagsL, tagsR, tagsU, tagsD = tags

    comm.Barrier()         # start MPI timer
    t_start = MPI.Wtime()

    # loop through time
    for j in range(1, Nt):
        u = MPI_Func(u, rank, px, col, row, tagsL, tagsR, tagsU, tagsD,
                     rankL, rankR, rankU, rankD)
        u = Updater(u, C, Kx, Ky)

    comm.Barrier()
    t_total = (MPI.Wtime() - t_start)  # stop MPI timer
    return t_total, None


def solver_2D_helper_g(u, ranks, col, row, tags, params):
    p,  px, py = params.p_vars
    C,  Kx, Ky = params.consts
    nx, ny, Nt = params.nx, params.ny, params.Nt
    Updater, MPI_Func = params.updater, params.mpi_func

    rank, rankL, rankR, rankU, rankD = ranks
    tagsL, tagsR, tagsU, tagsD = tags
    ug, U = create_global_vars(rank, params)

    comm.Barrier()         # start MPI timer
    t_start = MPI.Wtime()

    # loop through time
    for j in range(1, Nt):
        u = MPI_Func(u, rank, px, col, row, tagsL, tagsR, tagsU, tagsD,
                     rankL, rankR, rankU, rankD)
        u = Updater(u, C, Kx, Ky)

        # Gather parallel vectors to a serial vector
        comm.Gather(u[1:-1, 1:-1].flatten(), ug, root=0)
        if rank == 0:
            U[:, :, j] = get_Uj(u, ug, p, nx, ny)

    comm.Barrier()
    t_total = (MPI.Wtime() - t_start)  # stop MPI timer
    return t_total, U


def get_Uj(u, ug, p, nx, ny):
    # evenly split ug into a list of p parts
    temp = np.array_split(ug, p)
    # reshape each part
    temp = [a.reshape(ny, nx) for a in temp]
    return np.hstack(temp)


def create_global_vars(rank, params):
    x0, xf, dx, Nx, nx = params.x_vars
    y0, yf, dy, Ny, ny = params.y_vars
    nx, ny, Nt = params.nx, params.ny, params.Nt
    p = params.p
    f = params.ics_func

    if rank == 0:
        xg = np.linspace(x0 - dx/2, xf + dx/2, Nx+2)
        yg = np.linspace(y0 - dy/2, yf + dy/2, Ny+2)
        ug = np.array([f(xg, j) for j in yg])[1:-1, 1:-1].flatten()
        temp = np.array_split(ug, p)
        temp = [a.reshape(ny, nx) for a in temp]

        U = np.empty((ny, p*nx, Nt), dtype=np.float64)
        U[:, :, 0] = np.hstack(temp)
    else:
        ug = None
        U  = None

    return ug, U


class Params(object):
    """Placeholder for several constants and what not."""
    def __init__(self, x_vars, y_vars, t_vars, p_vars, consts, funcs):
        super(Params, self).__init__()
        self.x_vars = x_vars
        self.y_vars = y_vars
        self.t_vars = t_vars
        self.p_vars = p_vars
        self.consts = consts
        self.funcs  = funcs

        self.x0, self.xf, self.dx, self.Nx, self.nx = x_vars
        self.y0, self.yf, self.dy, self.Ny, self.ny = y_vars
        self.t0, self.tf, self.dt, self.Nt = t_vars
        self.p,  self.px, self.py = p_vars
        self.C,  self.Kx, self.Ky = consts
        self.ics_func, self.bcs_func, self.updater, self.mpi_func = funcs

    def set_x_vars(self, x_vars):
        self.x_vars = x_vars
        self.x0, self.xf, self.dx, self.Nx, self.nx = x_vars

    def set_y_vars(self, y_vars):
        self.y_vars = y_vars
        self.y0, self.yf, self.dy, self.Ny, self.ny = y_vars

    def set_t_vars(self, t_vars):
        self.t_vars = t_vars
        self.t0, self.tf, self.dt, self.Nt = t_vars

    def set_p_vars(self, p_vars):
        self.p_vars = p_vars
        self.p, self.px, self.py = p_vars

    def set_consts(self, consts):
        self.consts = consts
        self.C, self.Kx, self.Ky = consts

    def set_funcs(self, funcs):
        self.funcs = funcs
        self.ics_func, self.bcs_func, self.updater, self.mpi_func = funcs
