from __future__ import division
from heat_2d_setup import np, sys, MPI, comm, set_mpi_bdr2D, update_u, \
                          force_BCs_2D, animator_2D

"""
THIS FUNCTION IS NOT YET WORKING. Only the columns are passed through MPI right
now. The animation produced at the end doesn't look even close to correct. The
MPI code only sends columns, not both columns and rows, due to a deadlock issue.
This will be sorted out tomorrow (April 7th).
"""


# initial condition function
def f(x, y):
    # x, y can be arrays
    return np.sin(np.pi*x) * np.sin(np.pi*y)


def main(Updater, sc):
    # number of spatial points
    Nx = 128*sc
    Ny = 128*sc

    rank = comm.Get_rank()   # this process' ID
    p  = comm.Get_size()     # number of processors
    p2 = int(np.sqrt(p))     # number of processors in each direction
    nx = Nx/p2   # x-grid points per process
    ny = Ny/p2   # y-grid points per process

    indices = [(i, j) for i in xrange(p2) for j in xrange(p2)]
    procs = np.array_split(np.arange(p), p2)
    procs.reverse()
    procs = np.vstack(procs)                      # location of process blocks
    locs  = dict(zip(procs.flatten(), indices))   # map rank to location

    left  = np.roll(procs, 1,  1)
    right = np.roll(procs, -1, 1)
    up    = np.roll(procs, 1,  0)
    down  = np.roll(procs, -1, 0)

    # total number of points
    N = (Nx+2) * (Ny+2)

    # x conditions
    x0 = 0                  # start
    xf = 1                  # end
    dx = (xf-x0)/(Nx+1)     # spatial step size
    # this takes the interval [x0,xf] and splits it equally among all processes
    x = np.linspace(x0 + (rank % p2)*(xf-x0)/p2, x0 + (rank % p2 + 1)*(xf-x0)/p2, nx+2)

    # y conditions
    y0 = 0
    yf = 1
    dy = (yf-y0)/(Ny+1)
    y  = np.linspace(y0 + (rank // p2)*(yf-y0)/p2, y0 + (rank // p2 + 1)*(yf-y0)/p2, ny+2)

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

    comm.Barrier()         # start MPI timer
    t_start = MPI.Wtime()

    # loop through time
    for j in range(1, Nt):
        u = set_mpi_bdr2D(u, rank, p, p2, nx, ny, col, row, tagsL, tagsR,
                          tagsU, tagsD, left, right, up, down, locs)
        u = Updater(u, nx, ny, C, Kx, Ky)
        u = force_BCs_2D(u, rank, p, p2, nx, ny)

        # Gather parallel vectors to a serial vector
        comm.Gather(u[1:ny+1, 1:nx+1].flatten(), ug, root=0)
        if rank == 0:
            # evenly split ug into a list of p parts
            temp = np.array_split(ug, p)
            # reshape each part
            temp = [a.reshape(ny, nx) for a in temp]
            U[:, :, j] = np.hstack(temp)

    comm.Barrier()
    t_final = (MPI.Wtime() - t_start)  # stop MPI timer

    if rank == 0:
        # PLOTTING
        animator_2D(U, xg, yg, nx, ny, Nt, p, p2)

    return t_final

    sys.exit()

main(update_u, 1)
