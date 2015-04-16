#!/usr/bin/env python
# heat_2d_setup.py

from __future__ import division
import sys
import numpy as np
import time
import os
import numba
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


@numba.jit('float64[:,:](float64[:,:], float64[:,:])')
def numba_add(M1, M2):
    return np.add(M1, M2)


@numba.jit('float64[:,:](float64, float64[:,:])')
def numba_scale(c, M):
    # multiply M by c
    return np.multiply(c, M)


@numba.jit('float64[:,:](float64[:,:], float64, float64, float64)')
def calc_u_numba(u, C, Kx, Ky):
    # I'm sorry
    u[1:-1, 1:-1] = numba_add(
                        numba_add(
                            numba_scale(C, u[1:-1:, 1:-1]),
                            numba_scale(Kx, numba_add(u[1:-1:, 0:-2], u[1:-1, 2:])),
                        ),
                        numba_scale(Ky, numba_add(u[0:-2,  1:-1], u[2:, 1:-1]))
                    )
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


def set_x_bdr(u, rank, px, py, col, row, tags, rankL, rankR, rankU, rankD):
    # Sends columns.
    tagsL, tagsR, tagsU, tagsD = tags
    col_block = rank % px

    # Send odd-numbered columns
    if col_block % 2:
        comm.Send(u[:, 1].flatten(),  dest=rankL, tag=tagsL[rank])
        comm.Send(u[:, -2].flatten(), dest=rankR, tag=tagsR[rank])

    # Receive odd-numbered columns, send even-numbered columns
    else:
        comm.Recv(col, source=rankR, tag=tagsL[rankR])    # column from right
        u[:, -1] = col
        comm.Recv(col, source=rankL, tag=tagsR[rankL])    # column from left
        u[:, 0]  = col

        comm.Send(u[:, 1].flatten(),  dest=rankL, tag=tagsL[rank])
        comm.Send(u[:, -2].flatten(), dest=rankR, tag=tagsR[rank])

    # Receive even-numbered columns
    if col_block % 2:
        comm.Recv(col, source=rankR, tag=tagsL[rankR])    # column from right
        u[:, -1] = col
        comm.Recv(col, source=rankL, tag=tagsR[rankL])    # column from left
        u[:, 0]  = col

    return u


def set_y_bdr(u, rank, px, py, col, row, tags, rankL, rankR, rankU, rankD):
    # Sends rows
    tagsL, tagsR, tagsU, tagsD = tags
    row_block = rank // px

    # Send odd-numbered rows
    if row_block % 2:
        comm.Send(u[ 1, :].flatten(), dest=rankU, tag=tagsU[rank])
        comm.Send(u[-2, :].flatten(), dest=rankD, tag=tagsD[rank])

    # Receive odd-numbered rows, send even-numbered rows
    else:
        comm.Recv(row, source=rankD, tag=tagsU[rankD])    # row from below
        u[-1, :] = row
        comm.Recv(row, source=rankU, tag=tagsD[rankU])    # row from above
        u[0, :]  = row

        comm.Send(u[ 1, :].flatten(), dest=rankU, tag=tagsU[rank])
        comm.Send(u[-2, :].flatten(), dest=rankD, tag=tagsD[rank])

    # Receive even-numbered rows
    if row_block % 2:
        comm.Recv(row, source=rankD, tag=tagsU[rankD])    # row from below
        u[-1, :] = row
        comm.Recv(row, source=rankU, tag=tagsD[rankU])    # row from above
        u[0, :]  = row

    return u


def set_mpi_bdr2D(u, rank, px, py, col, row, tags, rankL, rankR, rankU, rankD):
    # get location (row, col) of this rank's block
    u = set_x_bdr(u, rank, px, py, col, row, tags, rankL, rankR, rankU, rankD)
    u = set_y_bdr(u, rank, px, py, col, row, tags, rankL, rankR, rankU, rankD)

    return u


def writer(t_total, method, sc, opt=None):
    if opt:
        filename = './tests/%s/%s/sc-%d.txt' % (method, opt, sc)
    else:
        filename = './tests/%s/sc-%d.txt' % (method, sc)

    # check to see if file exists; if it doesn't, create it.
    if not os.path.exists(filename):
        open(filename, 'a').close()

    # write time to the file
    F = open(filename, 'a')
    F.write('%f\n' % t_total)
    F.close()


def get_ims_x(U, xg, yg, Nt):
    ims = []
    for j in xrange(Nt):
        ims.append((plt.pcolormesh(xg[1:-1], yg[1:-1], U[:, :, j], norm=plt.Normalize(0, 1)), ))
    return ims


def get_ims_y(U, xg, yg, Nt, p):
    ims = []
    for j in xrange(Nt):
        U_j = np.vstack([arr.transpose() for arr in np.array_split(U[:, :, j].transpose(), p)])
        ims.append((plt.pcolormesh(xg[1:-1], yg[1:-1], U_j, norm=plt.Normalize(0, 1)), ))
    return ims


def get_ims_xy(U, xg, yg, nx, ny, Nt, p, px, py):
    ims = []
    for j in xrange(Nt):
        U_j = U[:, :, j].reshape(ny, p*nx)
        temp = [None for i in xrange(py)]
        for i in xrange(py):
            temp[i] = U_j[:, i*px*nx : (i+1)*px*nx]

        U_j = np.vstack(temp)
        ims.append((plt.pcolormesh(xg[1:-1], yg[1:-1], U_j, norm=plt.Normalize(0, 1)), ))
    return ims


def The_Animator(U, xg, yg, nx, ny, Nt, method, p, px, py):
    fig = plt.figure()
    print 'creating meshes...'
    if px == 1:
        ims = get_ims_y(U, xg, yg, Nt, p)
    elif py == 1:
        ims = get_ims_x(U, xg, yg, Nt)
    else:
        ims = get_ims_xy(U, xg, yg, nx, ny, Nt, p, px, py)
    print 'done creating meshes, attempting to put them together...'
    print 'saving...'
    im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000, blit=False)
    im_ani.save('./anims/MPI_SUPER_%s_%dpx_%dpy.mp4' % (method, px, py))
    print 'saved.'
