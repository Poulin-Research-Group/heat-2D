from heat_2d_setup import *
from _heatFortran import heatf

if rank == 0:
    xg = np.linspace(x0, xf, Mx+2)
    ug = np.array([f(xg, j) for j in y])[1:-1, 1:-1].flatten()
    t  = np.linspace(t0, tf, N)
else:
    ug = None

comm.Barrier()         # start MPI timer
t_start = MPI.Wtime()

# Loop over time
for j in range(1, N):

    # Send u[:, 1] to ID-1
    if 0 < rank:
        comm.Send(u[:, 1].flatten(), dest=rank-1, tag=tags[rank])

    # Receive u[mx+1] to ID+1
    if rank < p-1:
        comm.Recv(col, source=rank+1, tag=tags[rank+1])
        u[:, mx+1] = col

    # Send u[mx] to ID+1
    if rank < p-1:
        comm.Send(u[:, mx].flatten(), dest=rank+1, tag=tags[rank])

    # Receive u[0] to ID-1
    if 0 < rank:
        comm.Recv(col, source=rank-1, tag=tags[rank-1])
        u[:, 0] = col

    un = heatf(un, u, C, Kx, Ky)

    # Force Boundary Conditions
    un[0, :]    = 0.0  # first row
    un[My+1, :] = 0.0  # last row
    if rank == 0:
        un[:, 0]    = 0.0  # first col
    elif rank == p-1:
        un[:, mx+1] = 0.0  # last col

    # save solution
    # U[:,:,j] = un

    # Update solution
    u = un

comm.Barrier()
t_final = (MPI.Wtime() - t_start)  # stop MPI timer

comm.Gather(u[1:My+1, 1:mx+1].flatten(), ug, root=0)
if rank == 0:
    # evenly split ug into a list of p parts
    temp = np.array_split(ug, p)
    # reshape each part
    temp = [a.reshape(My, mx) for a in temp]
    U = np.zeros((My+2, Mx+2), dtype='d')
    U[1:My+1, 1:Mx+1] = np.hstack(temp)

# writer(t_final, u, writeToFile, i, 'fortran', 'par-step')

sys.exit()
