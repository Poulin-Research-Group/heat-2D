from heat_2d_setup import np, time, calc_u, BCs, writer, animator


# initial condition function
def f(x, y):
    # x, y can be arrays
    return np.sin(np.pi*x) * np.sin(np.pi*y)


def main(Updater, Force_BCs, sc):
    # number of spatial points
    Nx = 128*sc
    Ny = 128*sc

    # x conditions
    x0 = 0                       # start
    xf = 1                       # end
    dx = (xf-x0)/(Nx+1)          # spatial step size
    x = np.linspace(x0, xf, Nx+2)

    # y conditions
    y0 = 0
    yf = 1
    dy = (yf-y0)/(Ny+1)
    y  = np.linspace(y0, yf, Ny+2)

    # temporal conditions
    Nt = 1000
    t0 = 0             # start
    tf = 300           # end
    dt = (tf - t0)/Nt  # time step size
    t  = np.linspace(t0, tf, Nt)

    # coefficients
    # k  = 0.0002
    Kx = np.float64(0.02)                # PDE coeff for x terms
    Ky = np.float64(0.01)
    C  = 1 - 2*(Kx + Ky)

    # initial and global solutions
    u = np.array([f(x, j) for j in y])
    U = np.empty((Ny, Nx, Nt), dtype='float64')
    U[:, :, 0] = u[1:Ny+1, 1:Nx+1]

    t_start = time.time()

    # Loop over time
    for j in range(1, Nt):
        u = Updater(u, Nx, Ny, C, Kx, Ky)
        u = Force_BCs(u, Nx, Ny)
        U[:, :, j] = u[1:Ny+1, 1:Nx+1]

    t_final = time.time()

    # PLOTTING ======================================================
    animator(U, x, y, Nt)

    return t_final - t_start

main(calc_u, BCs, 1)
