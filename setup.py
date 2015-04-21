#!/usr/bin/env python
# heat_2d_setup.py

from __future__ import division
import sys
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from heatFortran import heatf
from heatFortran90 import heatf as heatf90
from mpi4py import MPI
comm = MPI.COMM_WORLD


def create_x(px, rank, x0, xf, dx, nx, Nx):
    col = rank % px
    xg = np.linspace(x0 - dx/2, xf + dx/2, Nx+2)
    x  = list(np.array_split(xg[1:-1], px)[col])
    return np.array([xg[col*nx]] + x + [xg[(col+1)*nx + 1]])


def create_y(px, py, rank, y0, yf, dy, ny, Ny):
    row = rank // px
    yg = np.linspace(y0 - dy/2, yf + dy/2, Ny+2)
    y  = list(np.array_split(yg[1:-1], py)[row])
    return np.array([yg[row*ny]] + y + [yg[(row+1)*ny + 1]])


def calc_u(u, C, Kx, Ky):
    u[1:-1, 1:-1] = C*u[1:-1, 1:-1]  + \
                        Kx*(u[1:-1, 0:-2] + u[1:-1, 2:])  + \
                        Ky*(u[0:-2, 1:-1] + u[2:, 1:-1])
    return u


def BCs(u, rank, p, px, py):
    u[ 0, :] = u[-2, :]  # first row
    u[-1, :] = u[ 1, :]  # last row
    u[:,  0] = u[:, -2]  # first col
    u[:, -1] = u[:,  1]  # last col
    return u


def BCs_X(u, rank, p, px, py):
    # set X boundary conditions (cols)
    u[:,  0] = u[:, -2]  # first col
    u[:, -1] = u[:,  1]  # last col

    return u


def BCs_Y(u, rank, p, px, py):
    # set Y boundary conditions (rows)
    u[ 0, :] = u[-2, :]   # first row
    u[-1, :] = u[ 1, :]   # last row

    return u


def BCs_XY(u, rank, p, px, py):
    # place holder to do nothing, as periodic BCs are already put in place via
    # MPI functions
    return u


def serial_bdr(u, rank, px, py, col, row, tags, rankL, rankR, rankU, rankD):
    # placeholder to do nothing, as serial solutions have the periodic BCs set
    # via the BCs function
    return u


METHODS   = ['numpy', 'f2py77', 'f2py90']
UPDATERS  = [calc_u, heatf, heatf90]


def solver_x(u, ranks, px, py, col, tags, params, save_solution):
    """
    Solves the 2D Heat Equation if it is parallelized only in x. If
    save_solution is True, then solver_x_helper_g is called. Otherwise,
    solver_x_helper is called.

    Returns
    -------
    tuple (float64, ndarray)   if save_solution is true
    tuple (float64, None)      if save_solution is false
    """

    if save_solution:
        t_total, U = solver_x_helper_g(u, ranks, col, tags, params)
    else:
        t_total, U = solver_x_helper(u, ranks, col, tags, params)


def solver_x_helper(u, ranks, col, tags, params):
    p,  px, py = params.p_vars
    C,  Kx, Ky = params.consts
    Updater, Set_BCs = params.updater, params.bcs_func
    Nt = params.Nt

    rank, rankL, rankR, rankU, rankD = ranks
    tagsL, tagsR, tagsU, tagsD = tags

    comm.Barrier()         # start MPI timer
    t_start = MPI.Wtime()

    # loop through time
    for j in range(1, Nt):
        u = set_x_bdr(u, rank, px, col, tagsL, tagsR, rankL, rankR)
        u = Updater(u, C, Kx, Ky)
        u = Set_BCs(u, rank, p, px, py)

    comm.Barrier()
    t_total = (MPI.Wtime() - t_start)  # stop MPI timer
    return t_total, None


def solver_x_helper_g(u, ranks, col, tags, params):
    p,  px, py = params.p_vars
    C,  Kx, Ky = params.consts
    nx, Ny, Nt = params.nx, params.ny, params.Nt
    Updater, Set_BCs = params.updater, params.bcs_func

    rank, rankL, rankR, rankU, rankD = ranks
    tagsL, tagsR, tagsU, tagsD = tags
    ug, U = create_global_vars(rank, params)

    comm.Barrier()         # start MPI timer
    t_start = MPI.Wtime()

    # loop through time
    for j in range(1, Nt):
        u = set_x_bdr(u, rank, px, col, tagsL, tagsR, rankL, rankR)
        u = Updater(u, C, Kx, Ky)
        u = Set_BCs(u, rank, p, px, py)

        # Gather parallel vectors to a serial vector
        comm.Gather(u[1:-1, 1:-1].flatten(), ug, root=0)
        if rank == 0:
            U[:, :, j] = get_Uj(u, ug, p, nx, Ny)

    comm.Barrier()
    t_total = (MPI.Wtime() - t_start)  # stop MPI timer
    return t_total, None


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
        self.ics_func, self.bcs_func, self.updater = funcs
