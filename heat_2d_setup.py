#!/usr/bin/env python
# heat_2d_stepping_mpi.py
#
# This will solve the 2-D heat equation in parallel using mpi4py

from __future__ import division
import sys
from mpi4py import MPI
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
comm = MPI.COMM_WORLD


def update_u(u, Nx, Ny, C, Kx, Ky):
    u[1:Ny+1, 1:Nx+1] = C*u[1:Ny+1, 1:Nx+1] + \
                        Kx*(u[1:Ny+1, 0:Nx] + u[1:Ny+1, 2:Nx+2]) + \
                        Ky*(u[0:Ny,  1:Nx+1] + u[2:Ny+2, 1:Nx+1])

    # Force Boundary Conditions
    u[0, :]    = 0.0  # first row
    u[Ny+1, :] = 0.0  # last row
    u[:, 0]    = 0.0  # first col
    u[:, Nx+1] = 0.0  # last col

    return u


def force_BCs(u, rank, p, nx, Ny):
    # Force Boundary Conditions
    if rank == 0:
        u[:, 0]    = 0.0  # first col
    elif rank == p-1:
        u[:, nx+1] = 0.0  # last col

    u[0, :]    = 0.0  # first row
    u[Ny+1, :] = 0.0  # last row

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

    # Force Boundary Conditions
    if rank == 0:
        u[:, 0]    = 0.0  # first col
    elif rank == p-1:
        u[:, nx+1] = 0.0  # last col

    u[0, :]    = 0.0  # first row
    u[Ny+1, :] = 0.0  # last row

    return u


def set_mpi_bdr2D(u, rank, p, p2, nx, ny, col, row, tagsL, tagsR, tagsU, tagsD, left, right, up, down, locs):
    # get location (row, col) of this rank's block
    loc = locs[rank]
    rankL = left[loc]
    rankR = right[loc]
    rankU = up[loc]
    rankD = down[loc]

    # Send odd-numbered columns
    if rank % 2:
        comm.Send(u[:, 1].flatten(),  dest=rankL, tag=tagsL[rank])
        comm.Send(u[:, nx].flatten(), dest=rankR, tag=tagsR[rank])

    # Receive odd-numbered columns, send even-numbered columns
    else:
        # receive cols
        comm.Recv(col, source=rankR, tag=tagsL[rankR])    # column from right
        u[:, nx+1] = col
        comm.Recv(col, source=rankL, tag=tagsR[rankL])    # column from left
        u[:, 0] = col

        # send cols
        comm.Send(u[:, 1].flatten(),  dest=rankL, tag=tagsL[rank])
        comm.Send(u[:, nx].flatten(), dest=rankR, tag=tagsR[rank])

    # Receive even-numbered columns, send odd-numbered rows
    if rank % 2:
        # receive cols
        comm.Recv(col, source=right[loc], tag=tagsL[rankR])    # column from right
        u[:, nx+1] = col
        comm.Recv(col, source=left[loc],  tag=tagsR[rankL])    # column from left
        u[:, 0] = col

    """
        # send rows
        comm.Send(u[1, :].flatten(),  dest=rankU, tag=tagsU[rank])
        comm.Send(u[ny, :].flatten(), dest=rankD, tag=tagsD[rank])
        print 'sent odd rows'

    # Receive odd-numbered rows, send even-numbered rows
    else:
        # receive rows
        comm.Recv(row, source=rankD, tag=tagsU[rankD])    # row from below
        u[ny+1, :] = row
        print 'RECVD ODD ROWS --------'
        comm.Recv(row, source=rankU, tag=tagsD[rankU])    # row from above
        u[0, :] = row

        # send rows
        comm.Send(u[1, :].flatten(),  dest=rankU, tag=tagsU[rank])
        comm.Send(u[ny, :].flatten(), dest=rankD, tag=tagsD[rank])
        print 'send even rows'

    # Receive even-numbered rows
    if rank % 2:
        comm.Recv(row, source=rankD, tag=tagsU[rankD])    # row from below
        u[ny+1, :] = row
        print 'received row 1'
        comm.Recv(row, source=rankU, tag=tagsD[rankU])    # row from above
        u[0, :] = row
        print 'recvd even rows '
    """

    # Force Boundary Conditions
    if rank >=  p - p2:
        u[0, :]    = 0.0  # first row
    elif rank < p2:
        u[ny+1, :] = 0.0  # last row

    if not rank % p2:
        u[:, 0]    = 0.0  # first col
    elif rank % p2 == p2-1:
        u[:, nx+1] = 0.0  # last col

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
    im_ani.save('stepping_mpi_%d.mp4' % p)
    # plt.show()
    print 'saved.'


def animator_2D(U, xg, yg, nx, ny, Nt, p, p2):
    fig = plt.figure()
    ims = []
    for j in xrange(Nt):
        U_j = U[:, :, j].reshape(ny, p*nx)
        temp = [None for i in xrange(p2)]
        for i in xrange(p2):
            temp[i] = U_j[:, i*p2*nx : (i+1)*p2*nx]

        U_j = np.vstack(temp)

        ims.append((plt.pcolormesh(xg[1:-1], yg[1:-1], U_j, norm=plt.Normalize(0, 1)), ))

    im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000, blit=False)
    im_ani.save('stepping_mpi_super_%d.mp4' % p)
