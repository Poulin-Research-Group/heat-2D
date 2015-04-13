from __future__ import division
from setup import np, sys, MPI, comm, set_mpi_bdr2D, calc_u, calc_u_numba, heatf, \
                  BCs_MPI_X, BCs_MPI_Y, BCs_MPI_XY, The_Animator, set_x_bdr,        \
                  set_y_bdr, plt


# initial condition function
def f(x, y):
    # x, y can be arrays
    return np.sin(np.pi*x) * np.sin(np.pi*y)


def main(Updater, sc=1, px=2, py=2):
    # number of spatial points
    Nx = 128*sc
    Ny = 128*sc

    # allow user to input number of cores dedicated to x-axis and y-axis,
    # then check to see if this is more than the actual number of cores
    p = px * py
    if p != comm.Get_size():
        raise Exception("Incorrect number of cores used; MPI is being run with %d, but %d was inputted." % (comm.Get_size(), p))

    rank = comm.Get_rank()   # this process' ID
    nx = Nx/px   # x-grid points per process
    ny = Ny/py   # y-grid points per process

    if px == 1:
        Set_MPI_Boundaries = set_y_bdr
        Force_BCs = BCs_MPI_Y
    elif py == 1:
        Set_MPI_Boundaries = set_x_bdr
        Force_BCs = BCs_MPI_X
    else:
        Set_MPI_Boundaries = set_mpi_bdr2D
        Force_BCs = BCs_MPI_XY

    indices = [(i, j) for i in xrange(py) for j in xrange(px)]
    procs = list(np.arange(p).reshape(py, px))
    procs.reverse()
    procs = np.array(procs)                       # location of process blocks
    locs  = dict(zip(procs.flatten(), indices))   # map rank to location
    loc   = locs[rank]

    left  = np.roll(procs, 1,  1)
    right = np.roll(procs, -1, 1)
    up    = np.roll(procs, 1,  0)
    down  = np.roll(procs, -1, 0)
    rankL = left[loc]
    rankR = right[loc]
    rankU = up[loc]
    rankD = down[loc]

    # x conditions
    x0 = 0                  # start
    xf = 1                  # end
    dx = (xf-x0)/(Nx+1)     # spatial step size
    # this takes the interval [x0,xf] and splits it equally among all processes
    x = np.linspace(x0 + (rank % px)*(xf-x0)/px, x0 + (rank % px + 1)*(xf-x0)/px, nx+2)

    # y conditions
    y0 = 0
    yf = 1
    dy = (yf-y0)/(Ny+1)
    y  = np.linspace(y0 + (rank // px)*(yf-y0)/py, y0 + (rank // px + 1)*(yf-y0)/py, ny+2)
    # for i in xrange(p):
    #     if i == rank:
    #         print y
    #     comm.Barrier()
    # return

    # temporal conditions
    Nt  = 1000         # time steps
    t0 = 0             # start
    tf = 300           # end
    dt = (tf - t0)/Nt  # time step size
    t  = np.linspace(t0, tf, Nt)

    # coefficients
    k  = 0.0002
    Kx = 0.02                # PDE coeff for x terms
    Ky = 0.01
    C  = 1 - 2*(Kx + Ky)

    # BUILD ZE GRID
    u   = np.array([f(x, j) for j in y])     # process' slice of soln
    col = np.empty(ny+2, dtype='d')
    row = np.empty(nx+2, dtype='d')

    # define global variables (across all processes)
    if rank == 0:
        xg = np.linspace(x0, xf, Nx+2)
        yg = np.linspace(y0, yf, Ny+2)
        ug = np.array([f(xg, j) for j in yg])[1:-1, 1:-1].flatten()
        temp = np.array_split(ug, p)
        temp = [a.reshape(ny, nx) for a in temp]

        U = np.empty((ny, p*nx, Nt), dtype=np.float64)
        U[:, :, 0] = np.hstack(temp)
    else:
        ug = None

    tagsL = dict([(j, j+1) for j in xrange(p)])
    tagsR = dict([(j,   p + (j+1)) for j in xrange(p)])
    tagsU = dict([(j, 2*p + (j+1)) for j in xrange(p)])
    tagsD = dict([(j, 3*p + (j+1)) for j in xrange(p)])
    tags  = (tagsL, tagsR, tagsU, tagsD)

    comm.Barrier()         # start MPI timer
    t_start = MPI.Wtime()

    # loop through time
    for j in range(1, Nt):
        u = Set_MPI_Boundaries(u, rank, px, py, col, row, tags, rankL, rankR, rankU, rankD, loc)
        u = Updater(u, C, Kx, Ky)
        u = Force_BCs(u, rank, p, px, py)

        # Gather parallel vectors to a serial vector
        comm.Gather(u[1:-1, 1:-1].flatten(), ug, root=0)
        if rank == 0:
            # evenly split ug into a list of p parts
            temp = np.array_split(ug, p)
            # reshape each part
            temp = [a.reshape(ny, nx) for a in temp]
            U[:, :, j] = np.hstack(temp)

            U_final = U[:, :, j].reshape(ny, p*nx)
            temp = [None for i in xrange(py)]
            for i in xrange(py):
                temp[i] = U_final[:, i*px*nx : (i+1)*px*nx]

            U_final = np.vstack(temp)
            plt.title('time step: ' + str(j))
            plt.pcolormesh(xg[1:-1], yg[1:-1], U_final, norm=plt.Normalize(0, 1))
            plt.colorbar()
            plt.show()

    comm.Barrier()
    t_final = (MPI.Wtime() - t_start)  # stop MPI timer

    # PLOTTING
    if rank == 0:
        if Updater is calc_u:
            method = 'numpy'
        elif Updater is heatf:
            method = 'f2py-f77'
        elif Updater is calc_u_numba:
            method = 'numba'
        The_Animator(U, xg, yg, nx, ny, Nt, method, p, px, py)
        # U_final = U[:, :, -1].reshape(ny, p*nx)
        # temp = [None for i in xrange(py)]
        # for i in xrange(py):
        #     temp[i] = U_final[:, i*px*nx : (i+1)*px*nx]

        # U_final = np.vstack(temp)
        # plt.pcolormesh(xg[1:-1], yg[1:-1], U_final, norm=plt.Normalize(0, 1))
        # plt.colorbar()
        # plt.show()

    return t_final


# main(calc_u, 1, 2, 2)
# main(calc_u_numba, 1, 2, 2)
# main(heatf, 1, 2, 2)          # px = 2, py = 2
main(heatf, 1, 1, 4)          # px = 1, py = 4
# main(heatf, 1, 4, 1)          # px = 4, py = 1
