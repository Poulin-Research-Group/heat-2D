from heat_2d_setup import *
from _heatFortran import heatf

t_start = time.time()

# Loop over time
for j in range(1, N):
    un = heatf(un, u, C, Kx, Ky)

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

writer(t_final, u, writeToFile, i, 'fortran', 'ser-step')

sys.exit()
