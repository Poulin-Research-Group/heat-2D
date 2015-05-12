from __future__ import division
import numpy as np
import os
from mpi4py import MPI
comm = MPI.COMM_WORLD


def create_ranks(rank, p, px, py):
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
    return rank, rankL, rankR, rankU, rankD


def create_tags(p):
    tagsL = dict([(j, j+1) for j in xrange(p)])
    tagsR = dict([(j,   p + (j+1)) for j in xrange(p)])
    tagsU = dict([(j, 2*p + (j+1)) for j in xrange(p)])
    tagsD = dict([(j, 3*p + (j+1)) for j in xrange(p)])
    return tagsL, tagsR, tagsU, tagsD


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


def send_cols_periodic(u, rank, px, col, tagsL, tagsR, rankL, rankR):
    # Sends columns.
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


def send_rows_periodic(u, rank, px, row, tagsU, tagsD, rankU, rankD):
    # Sends rows
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


def send_periodic(u, rank, px, col, row, tagsL, tagsR, tagsU, tagsD,
                     rankL, rankR, rankU, rankD):
    u = send_cols_periodic(u, rank, px, col, tagsL, tagsR, rankL, rankR)
    u = send_rows_periodic(u, rank, px, row, tagsU, tagsD, rankU, rankD)
    return u
