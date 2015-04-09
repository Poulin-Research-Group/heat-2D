#!/usr/bin/env python
# heat_2d_setup.py

from __future__ import division
import sys
from mpi4py import MPI
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
comm = MPI.COMM_WORLD


def calc_u(u, Nx, Ny, C, Kx, Ky):
    u[1:Ny+1, 1:Nx+1] = C*u[1:Ny+1, 1:Nx+1] + \
                        Kx*(u[1:Ny+1, 0:Nx] + u[1:Ny+1, 2:Nx+2]) + \
                        Ky*(u[0:Ny,  1:Nx+1] + u[2:Ny+2, 1:Nx+1])
    return u


def BCs(u, Nx, Ny):
    u[0, :]    = 0.0  # first row
    u[Ny+1, :] = 0.0  # last row
    u[:, 0]    = 0.0  # first col
    u[:, Nx+1] = 0.0  # last col
    return u


def BCs_MPI(u, rank, p, nx, Ny):
    # Force Boundary Conditions
    if rank == 0:
        u[:, 0]    = 0.0  # first col
    elif rank == p-1:
        u[:, nx+1] = 0.0  # last col

    u[0, :]    = 0.0  # first row
    u[Ny+1, :] = 0.0  # last row

    return u


def BCs_MPI_2D(u, rank, p, p2, nx, ny):
    # Force Boundary Conditions
    if rank >= p - p2:
        u[0, :]    = 0.0  # first row
    elif rank < p2:
        u[ny+1, :] = 0.0  # last row

    if not rank % p2:
        u[:, 0]    = 0.0  # first col
    elif rank % p2 == p2-1:
        u[:, nx+1] = 0.0  # last col

    return u


def set_mpi_bdr(u, rank, p, nx, Ny, col, tags):
    # Send u[:, 1] to ID-1
    if 0 < rank:
        comm.Send(u[:, 1].flatten(), dest=rank-1, tag=tags[rank])

    # Receive u[:, nx+1] to ID+1
    if rank < p-1:
        comm.Recv(col, source=rank+1, tag=tags[rank+1])
        u[:, nx+1] = col

    # Send u[:, nx] to ID+1
    if rank < p-1:
        comm.Send(u[:, nx].flatten(), dest=rank+1, tag=tags[rank])

    # Receive u[:, 0] to ID-1
    if 0 < rank:
        comm.Recv(col, source=rank-1, tag=tags[rank-1])
        u[:, 0] = col

    return u


def set_mpi_bdr2D(u, rank, p, p2, nx, ny, col, row, tagsL, tagsR, tagsU, tagsD, left, right, up, down, locs):
    # get location (row, col) of this rank's block
    loc = locs[rank]
    rankL = left[loc]
    rankR = right[loc]
    rankU = up[loc]
    rankD = down[loc]

    row_block = rank // p2
    col_block = rank % p2

    # Send odd-numbered columns
    if col_block % 2:
        comm.Send(u[:, 1].flatten(),  dest=rankL, tag=tagsL[rank])
        comm.Send(u[:, nx].flatten(), dest=rankR, tag=tagsR[rank])

    # Receive odd-numbered columns, send even-numbered columns
    else:
        comm.Recv(col, source=rankR, tag=tagsL[rankR])    # column from right
        u[:, nx+1] = col
        comm.Recv(col, source=rankL, tag=tagsR[rankL])    # column from left
        u[:, 0] = col

        comm.Send(u[:, 1].flatten(),  dest=rankL, tag=tagsL[rank])
        comm.Send(u[:, nx].flatten(), dest=rankR, tag=tagsR[rank])

    # Receive even-numbered columns
    if col_block % 2:
        comm.Recv(col, source=right[loc], tag=tagsL[rankR])    # column from right
        u[:, nx+1] = col
        comm.Recv(col, source=left[loc],  tag=tagsR[rankL])    # column from left
        u[:, 0] = col

    # Send odd-numbered rows
    if row_block % 2:
        comm.Send(u[1, :].flatten(),  dest=rankU, tag=tagsU[rank])
        comm.Send(u[ny, :].flatten(), dest=rankD, tag=tagsD[rank])

    # Receive odd-numbered rows, send even-numbered rows
    else:
        comm.Recv(row, source=rankD, tag=tagsU[rankD])    # row from below
        u[ny+1, :] = row
        comm.Recv(row, source=rankU, tag=tagsD[rankU])    # row from above
        u[0, :] = row

        comm.Send(u[1, :].flatten(),  dest=rankU, tag=tagsU[rank])
        comm.Send(u[ny, :].flatten(), dest=rankD, tag=tagsD[rank])

    # Receive even-numbered rows
    if row_block % 2:
        comm.Recv(row, source=rankD, tag=tagsU[rankD])    # row from below
        u[ny+1, :] = row
        comm.Recv(row, source=rankU, tag=tagsD[rankU])    # row from above
        u[0, :] = row

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


def animator(U, xg, yg, Nt, p=1):
    # PLOTTING
    fig = plt.figure()
    ims = []
    for j in xrange(Nt):
        ims.append((plt.pcolormesh(xg[1:-1], yg[1:-1], U[:, :, j], norm=plt.Normalize(0, 1)), ))

    print 'done creating meshes, attempting to put them together...'
    im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000, blit=False)

    print 'saving...'
    im_ani.save('./anims/stepping_mpi_%d.mp4' % p)
    # plt.show()
    print 'saved.'


def animator_2D(U, xg, yg, nx, ny, Nt, p, p2):
    fig = plt.figure()
    ims = []
    # plt.pcolormesh(xg[1:-1], yg[1:-1], U[:, :, 0])
    # print U[:, :, 0]
    # plt.show()
    # plt.pcolormesh(xg[1:-1], yg[1:-1], U[:, :, 1])
    # print U[:, :, 1]
    # return
    print 'creating meshes...'
    for j in xrange(Nt):
        U_j = U[:, :, j].reshape(ny, p*nx)
        temp = [None for i in xrange(p2)]
        for i in xrange(p2):
            temp[i] = U_j[:, i*p2*nx : (i+1)*p2*nx]

        U_j = np.vstack(temp)
        ims.append((plt.pcolormesh(xg[1:-1], yg[1:-1], U_j, norm=plt.Normalize(0, 1)), ))
    print 'done creating meshes, attempting to put them together...'

    print 'saving...'
    im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000, blit=False)
    im_ani.save('./anims/stepping_mpi_super_%d.mp4' % p)
    print 'saved.'
