#!/usr/bin/env python
# heat_2d_setup.py

from __future__ import division
import numpy as np
import time
import os
from heatFortran77 import heatf as heat_f77
from heatFortran90 import heatf as heat_f90
from mpi4py import MPI
from fjp_helpers.animator import mesh_animator
from fjp_helpers.bc import *
from fjp_helpers.mpi import *
from fjp_helpers.misc import *
comm = MPI.COMM_WORLD


def heat_np(u, C, Kx, Ky):
    u[1:-1, 1:-1] = C*u[1:-1, 1:-1]  + \
                        Ky*(u[1:-1, 0:-2] + u[1:-1, 2:])  + \
                        Kx*(u[0:-2, 1:-1] + u[2:, 1:-1])
    return u


METHODS   = ['numpy', 'f2py77', 'f2py90']
UPDATERS  = [heat_np, heat_f77, heat_f90]


def solver(Updater, params, px, py, SAVE_TIME=False, ANIMATE=False, SAVE_SOLN=False):

    p = px * py
    if p != comm.Get_size():
        if comm.Get_rank() == 0:
            raise Exception("Incorrect number of cores used; MPI is being run with %d, "
                            "but %d was inputted." % (comm.Get_size(), p))

    # update Params object with p value and updater
    params.set_p_vars([p, px, py])
    params.updater = Updater

    # create ranks and tags in all directions
    rank = comm.Get_rank()
    rankL, rankR, rankU, rankD = create_ranks(rank, p, px, py)[1:]
    tagsL, tagsR, tagsU, tagsD = create_tags(p)

    # get variables and type of BCs
    x0, xf, dx, Nx, nx = params.x_vars
    y0, yf, dy, Ny, ny = params.y_vars
    t0, tf, dt, Nt = params.t_vars
    C, Kx, Ky = params.consts
    bcs_type  = params.bcs_type

    # split x and y values along each process
    x = create_x(px, rank, x0, xf, dx, nx, Nx)
    y = create_y(px, py, rank, y0, yf, dy, ny, Ny)
    t = np.linspace(t0, tf, Nt)

    # get the initial condition function
    f = params.ics

    # BUILD ZE GRID (of initial conditions)
    xx, yy = np.meshgrid(x, y)
    u = f(xx, yy)

    # create ghost column and row
    col = np.empty(ny+2, dtype='d')
    row = np.empty(nx+2, dtype='d')

    # update Params object with BC functions
    # if the BCs are periodic...
    if bcs_type == 'P':
        # if any of the BC functions passed were None, use default periodic BC functions
        if any([bc is None for bc in params.bcs]):
            params.set_bc_funcs([set_periodic_BC, set_periodic_BC_x, set_periodic_BC_y,
                                 set_periodic_BC_placeholder])

    # if we have one process per direction, we're solving it in serial
    if px == 1 and py == 1:
        params.mpi_func = None
        params.bc_func  = params.bc_s
        t_total = solver_serial(u, params, ANIMATE, SAVE_SOLN)

    # if we have one process in y, we're parallelizing the solution in x
    elif py == 1:
        ranks = (rank,  rankL, rankR)
        tags  = (tagsL, tagsR)
        params.mpi_func = send_cols_periodic
        params.bc_func  = params.bc_y
        t_total = solver_mpi_1D(u, ranks, col, tags, params, ANIMATE, SAVE_SOLN)

    # if we have one process in x, we're parallelizing the solution in y
    elif px == 1:
        ranks = (rank,  rankU, rankD)
        tags  = (tagsU, tagsD)
        params.mpi_func = send_rows_periodic
        params.bc_func  = params.bc_x
        t_total = solver_mpi_1D(u, ranks, row, tags, params, ANIMATE, SAVE_SOLN)

    # otherwise we're parallelizing the solution in both directions
    else:
        ranks = (rank,  rankL, rankR, rankU, rankD)
        tags  = (tagsL, tagsR, tagsU, tagsD)
        params.mpi_func = send_periodic
        params.bc_func  = params.bc_xy
        t_total = solver_mpi_2D(u, ranks, col, row, tags, params, ANIMATE, SAVE_SOLN)

    if rank == 0:
        # save the time to a file
        if SAVE_TIME:
            filename = params.filename_time.split(os.sep)
            direc, filename = os.sep.join(filename[:-1]), filename[-1]
            if not os.path.isdir(direc):
                os.makedirs(direc)
            write_time(t_total, direc, filename)

        print t_total

    return t_total


def animate_solution(U, rank, params):
    x0, xf, dx, Nx, nx = params.x_vars
    y0, yf, dy, Ny, ny = params.y_vars
    p, px, py = params.p_vars
    Nt = params.Nt
    if rank == 0:
        xg = np.linspace(x0 - dx/2, xf + dx/2, Nx+2)
        yg = np.linspace(y0 - dy/2, yf + dy/2, Ny+2)
        filename = params.filename_anim.split(os.sep)    # split filename according to OS' separator
        direc, filename = os.sep.join(filename[:-1]), filename[-1]
        mesh_animator(U, xg, yg, nx, ny, Nt, p, px, py, direc, filename)


def solver_serial(u, params, animate, save_soln):
    """
    Solves the 2D Heat Equation in serial.

    Returns
    -------
    float64
        Total time taken for equation to be solved.
    """
    run_normal = True

    if animate:
        t_total, U = solver_serial_helper_g(u, params)
        animate_solution(U, 0, params)
        run_normal = False

    if save_soln:
        t_total = solver_serial_helper_w(u, params)
        run_normal = False

    if run_normal:
        t_total = solver_serial_helper(u, params)

    return t_total


def solver_serial_helper(u, params):
    C,  Kx, Ky = params.consts
    Updater, Set_BCs, = params.updater, params.bc_func
    Nt = params.Nt

    t_start = time.time()

    # loop through time
    for j in range(1, Nt):
        u = Set_BCs(u)
        u = Updater(u, C, Kx, Ky)

    t_total = (time.time() - t_start)
    return t_total


def solver_serial_helper_g(u, params):
    C,  Kx, Ky = params.consts
    Updater, Set_BCs, = params.updater, params.bc_func
    Nt = params.Nt

    U = create_global_vars(0, params)[1]

    t_start = time.time()

    # loop through time
    for j in range(1, Nt):
        u = Set_BCs(u)
        u = Updater(u, C, Kx, Ky)

        U[:, j] = u[1:-1, 1:-1].flatten()

    t_total = (time.time() - t_start)
    return t_total, U


def solver_serial_helper_w(u, params):
    Updater, Set_BCs, = params.updater, params.bc_func
    C, Kx, Ky = params.consts
    Nt = params.Nt
    f  = open(params.filename_soln, 'w')

    t_start = time.time()

    # loop through time
    for j in range(1, Nt):
        u = Set_BCs(u)
        u = Updater(u, C, Kx, Ky)

        np.savetxt(f, u[1:-1, 1:-1])
        f.write('# Next solution\n')

    t_total = (time.time() - t_start)
    return t_total


def solver_mpi_1D(u, ranks, ghost_arr, tags, params, animate, save_soln):
    run_normal = True

    if animate:
        t_total, U = solver_1D_helper_g(u, ranks, ghost_arr, tags, params)
        animate_solution(U, ranks[0], params)
        run_normal = False

    if save_soln:
        t_total = solver_1D_helper_w(u, ranks, ghost_arr, tags, params)
        run_normal = False

    if run_normal:
        t_total = solver_1D_helper(u, ranks, ghost_arr, tags, params)

    return t_total


def solver_1D_helper(u, ranks, ghost_arr, tags, params):
    Updater, Set_BCs, MPI_Func = params.updater, params.bc_func, params.mpi_func
    p, px, py = params.p_vars
    C, Kx, Ky = params.consts
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
    return t_total


def solver_1D_helper_g(u, ranks, ghost_arr, tags, params):
    Updater, Set_BCs, MPI_Func = params.updater, params.bc_func, params.mpi_func
    p, px, py = params.p_vars
    C, Kx, Ky = params.consts
    Nt = params.Nt

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
            U[:, j] = ug

    comm.Barrier()
    t_total = (MPI.Wtime() - t_start)  # stop MPI timer
    return t_total, U


def solver_1D_helper_w(u, ranks, ghost_arr, tags, params):
    Updater, Set_BCs, MPI_Func = params.updater, params.bc_func, params.mpi_func
    p,  px, py = params.p_vars
    C,  Kx, Ky = params.consts
    Nx, Ny, Nt = params.Nx, params.Ny, params.Nt
    nx, ny     = params.nx, params.ny
    f  = open(params.filename_soln, 'w')

    if px == 1:
        reshaper = reshape_soln_y
    else:
        reshaper = reshape_soln_x

    rank, rankLU, rankRD = ranks
    tagsLU, tagsRD = tags
    ug = np.empty(Ny*Nx, dtype='d')

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
            np.savetxt(f, reshaper(ug, nx, ny, p, px, py))
            f.write('# Next solution\n')

    comm.Barrier()
    t_total = (MPI.Wtime() - t_start)  # stop MPI timer
    return t_total


def solver_mpi_2D(u, ranks, col, row, tags, params, animate, save_soln):
    run_normal = True

    if animate:
        t_total, U = solver_2D_helper_g(u, ranks, col, row, tags, params)
        animate_solution(U, ranks[0], params)
        run_normal = False

    if save_soln:
        t_total = solver_2D_helper_w(u, ranks, col, row, tags, params)
        run_normal = False

    if run_normal:
        t_total = solver_2D_helper(u, ranks, col, row, tags, params)

    return t_total


def solver_2D_helper(u, ranks, col, row, tags, params):
    Updater, MPI_Func = params.updater, params.mpi_func
    p, px, py = params.p_vars
    C, Kx, Ky = params.consts
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
    return t_total


def solver_2D_helper_g(u, ranks, col, row, tags, params):
    Updater, MPI_Func = params.updater, params.mpi_func
    p, px, py = params.p_vars
    C, Kx, Ky = params.consts
    Nt = params.Nt

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
            U[:, j] = ug

    comm.Barrier()
    t_total = (MPI.Wtime() - t_start)  # stop MPI timer
    return t_total, U


def solver_2D_helper_w(u, ranks, ghost_arr, tags, params):
    Updater, MPI_Func = params.updater, params.mpi_func
    p,  px, py = params.p_vars
    C,  Kx, Ky = params.consts
    Nx, Ny, Nt = params.Nx, params.Ny, params.Nt
    nx, ny     = params.nx, params.ny
    f = open(params.filename_soln, 'w')

    rank,  rankL, rankR, rankU, rankD = ranks
    tagsL, tagsR, tagsU, tagsD = tags
    ug = np.empty(Ny*Nx, dtype='d')

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
            np.savetxt(f, reshape_soln_xy(ug, nx, ny, p, px, py))
            f.write('# Next solution\n')

    comm.Barrier()
    t_total = (MPI.Wtime() - t_start)  # stop MPI timer
    return t_total


def create_global_vars(rank, params):
    x0, xf, dx, Nx, nx = params.x_vars
    y0, yf, dy, Ny, ny = params.y_vars
    Nt = params.Nt
    f  = params.ics

    if rank == 0:
        xg = np.linspace(x0 - dx/2, xf + dx/2, Nx+2)
        yg = np.linspace(y0 - dy/2, yf + dy/2, Ny+2)
        xx, yy = np.meshgrid(xg, yg)
        ug = f(xx, yy)[1:-1, 1:-1].flatten()

        U = np.empty((Nx*Ny, Nt), dtype=np.float64)
        U[:, 0] = ug
    else:
        ug = None
        U  = None

    return ug, U


def set_periodic_BC_placeholder(u):
    return u


class Params(object):
    """Placeholder for several constants and what not."""
    def __init__(self):
        super(Params, self).__init__()
        self.x_vars, self.y_vars, self.t_vars, self.p_vars, self.consts = 5*[None]
        self.funcs = {}

    def __str__(self):
        return "x-vars: %s\ny-vars: %s\nt-vars: %s\np-vars: %s\nconsts: %s\n""" % (
            str(self.x_vars), str(self.y_vars), str(self.t_vars), str(self.p_vars),
            str(self.consts)
        )

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
        # funcs is a dictionary
        self.funcs = funcs
        self.ics = funcs['ic']
        self.bc_s, self.bc_x  = funcs['bc_s'], funcs['bc_x']
        self.bc_y, self.bc_xy = funcs['bc_y'], funcs['bc_xy']
        self.updater  = funcs['updater']
        self.mpi_func = funcs['mpi']

    def set_bc_funcs(self, bc_funcs):
        # bc_funcs is an array, [serial, x, y, xy]
        self.bc_s, self.bc_x, self.bc_y, self.bc_xy = bc_funcs
        self.funcs['bc_s'], self.funcs['bc_x']  = self.bc_s, self.bc_x
        self.funcs['bc_y'], self.funcs['bc_xy'] = self.bc_y, self.bc_xy
        self.bcs = bc_funcs
