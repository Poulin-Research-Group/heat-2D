from __future__ import division
from setup import np, sys, MPI, comm, set_mpi_bdr, calc_u, BCs, BCs_MPI, \
                  animator, heatf, calc_u_numba


# initial condition function
def f(x, y):
    # x, y can be arrays
    return np.sin(np.pi*x) * np.sin(np.pi*y)


def main(Updater, Force_BCs, sc):
    # number of spatial points
    Nx = 128*sc
    Ny = 128*sc

    rank = comm.Get_rank()   # this process' ID
    p = comm.Get_size()      # number of processors
    nx = Nx/p   # x-grid points per process

    # total number of points
    N = (Nx+2) * (Ny+2)

    # x conditions
    x0 = 0                       # start
    xf = 1                       # end
    dx = (xf-x0)/(Nx+1)          # spatial step size
    # this takes the interval [x0,xf] and splits it equally among all processes
    x = np.linspace(x0 + rank*(xf-x0)/p, x0 + (rank+1)*(xf-x0)/p, nx+2)

    # y conditions
    y0 = 0
    yf = 1
    dy = (yf-y0)/(Ny+1)
    y  = np.linspace(y0, yf, Ny+2)

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
    col = np.empty(Ny+2, dtype='d')

    # define global variables (across all processes)
    if rank == 0:
        xg = np.linspace(x0, xf, Nx+2)
        ug = np.array([f(xg, j) for j in y])[1:-1, 1:-1].flatten()
        U  = np.empty((Ny, Nx, Nt), dtype=np.float64)
        U[:, :, 0] = ug.reshape(Ny, Nx)
        t  = np.linspace(t0, tf, Nt)
    else:
        ug = None

    tags = dict([(j, j+5) for j in xrange(p)])

    comm.Barrier()         # start MPI timer
    t_start = MPI.Wtime()

    # loop through time
    for j in range(1, Nt):

        u = set_mpi_bdr(u, rank, p, nx, Ny, col, tags)
        u = Updater(u, C, Kx, Ky)
        u = Force_BCs(u, rank, p, nx, Ny)

        # Gather parallel vectors to a serial vector
        comm.Gather(u[1:Ny+1, 1:nx+1].flatten(), ug, root=0)
        if rank == 0:
            # evenly split ug into a list of p parts
            temp = np.array_split(ug, p)
            # reshape each part
            temp = [a.reshape(Ny, nx) for a in temp]
            U[:, :, j] = np.hstack(temp)

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
        animator(U, xg, y, Nt, method, p)

    return t_final

    sys.exit()

main(calc_u, BCs_MPI, 1)
main(calc_u_numba, BCs_MPI, 1)
main(heatf, BCs_MPI, 1)
