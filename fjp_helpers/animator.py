import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from misc import reshape_soln_x, reshape_soln_y, reshape_soln_xy


def mesh_animator(U, xg, yg, nx, ny, Nt, p, px, py, direc, filename):

    fig = plt.figure()
    print 'creating meshes...'

    if px == 1 and py == 1:
        reshaper = reshape_soln_x
    elif px == 1:
        reshaper = reshape_soln_y
    elif py == 1:
        reshaper = reshape_soln_x
    else:
        reshaper = reshape_soln_xy

    ims = []
    for j in xrange(Nt):
        ug = U[:, j]
        ims.append((plt.pcolormesh(xg[1:-1], yg[1:-1], reshaper(ug, nx, ny, p, px, py),
                    norm=plt.Normalize(0, 1)), ))

    print 'done creating meshes, attempting to put them together...'
    print 'saving...'
    im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000, blit=False)

    # check to see if directory exists; if it doesn't, create it.
    if not os.path.isdir(direc):
        os.makedirs(direc)

    filename = os.path.join(direc, filename)

    im_ani.save(filename)
    print 'saved.'
