from heat_2d_setup import *
from numba import jit
import numba


@numba.jit('float64[:,:](float64[:,:])')
def update(u):
    return C*u[1:My+1:, 1:mx+1]  + \
            Kx*(u[1:My+1:, 0:mx] + u[1:My+1, 2:mx+2]) + \
            Ky*(u[0:My,  1:mx+1] + u[2:My+2, 1:mx+1])


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
