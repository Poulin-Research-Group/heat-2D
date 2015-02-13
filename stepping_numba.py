from heat_2d_setup import *
from numba import jit
import numba

"""
TODO:
sometimes this code runs in about 0.74 seconds, compared to the normal code
which runs in about 1.05 seconds. that's a kickass improvement. However, it
doesn't always! I'll need to find a way to do AOT compiling with Numba to
try to get the results I want.
"""


@numba.jit('float64[:,:](float64[:,:], float64[:,:])')
def numba_add(M1, M2):
    return np.add(M1, M2)


@numba.jit('float64[:,:](float64, float64[:,:])')
def numba_scale(c, M):
    # multiply M by c
    return np.multiply(c, M)


@numba.jit('float64[:,:](float64[:,:])')
def update(u):
    # I'm sorry
    return numba_add(
             numba_add(
                numba_scale(C, u[1:My+1:, 1:mx+1]),
                numba_scale(Kx, numba_add(u[1:My+1:, 0:mx], u[1:My+1, 2:mx+2])),
             ),
             numba_scale(Ky, numba_add(u[0:My,  1:mx+1], u[2:My+2, 1:mx+1]))
            )


t_start = time.time()

# Loop over time
for j in range(1, N):
    un[1:My+1, 1:Mx+1] = update(u)

    # Force Boundary Conditions
    un[0, :]    = 0.0  # first row
    un[My+1, :] = 0.0  # last row
    un[:, 0]    = 0.0  # first col
    un[:, Mx+1] = 0.0  # last col

    # save solution
    # U[:,:,j] = un

    # Update solution
    u = un

t_final = time.time()
print t_final - t_start

writer(t_final, u, writeToFile, i, 'numba', 'ser-step')

sys.exit()

# PLOTTING
fig = plt.figure()
ims = []
for j in xrange(N):
    print j
    ims.append((plt.pcolormesh(x, y, U[:, :, j], norm=plt.Normalize(0, 1)), ))  # THIS IS A TUPLE

im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000, blit=False)

im_ani.save('im.mp4')
plt.show()

sys.exit()
