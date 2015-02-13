from heat_2d_setup import *

comm.Barrier()         # start MPI timer
t_start = MPI.Wtime()

# loop through time
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

    un[1:My+1, 1:mx+1] = C*u[1:My+1:, 1:mx+1] + \
                         Kx*(u[1:My+1:, 0:mx] + u[1:My+1, 2:mx+2]) + \
                         Ky*(u[0:My,  1:mx+1] + u[2:My+2, 1:mx+1])

    # Force Boundary Conditions
    if rank == 0:
        un[:, 0]    = 0.0  # first col
    elif rank == p-1:
        un[:, mx+1] = 0.0  # last col

    un[0, :]    = 0.0  # first row
    un[My+1, :] = 0.0  # last row

    # update soln
    u = un

    """
    # Gather parallel vectors to a serial vector
    comm.Gather(u[1:My+1, 1:mx+1].flatten(), ug, root=0)
    if rank == 0:
        # evenly split ug into a list of p parts
        temp = np.array_split(ug, p)
        # reshape each part
        temp = [a.reshape(My, mx) for a in temp]
        U[:, :, j] = np.hstack(temp)
    """

comm.Barrier()
t_final = (MPI.Wtime() - t_start)  # stop MPI timer


writer(t_final, u, writeToFile, i, 'original', 'par-step')

sys.exit()

if rank == 0:
    # PLOTTING
    fig = plt.figure()
    ims = []
    for j in xrange(N):
        ims.append((plt.pcolormesh(xg[1:-1], y[1:-1], U[:, :, j], norm=plt.Normalize(0, 1)), ))

    print 'done creating meshes, attempting to put them together...'
    im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000, blit=False)

    print 'saving...'
    im_ani.save('stepping_mpi_%d.mp4' % p)
    # plt.show()
    print 'saved.'
