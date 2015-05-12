import os
import numpy as np


def write_time(t_total, direc, filename):
    """
    Write the time to a file in a specific directory. Check to see if the
    directory exists first, and then if the file exists. If either of these are
    not true, then they are created.

    Parameters
    ----------
    t_total : float64
        Time. In our (read: my) cases, this is the amount of time it took for
        the equations to be solved numerically.
    direc : string
        The name of the directory that the file will be located in.
    filename : string
        The name of the file that the time is written to. This should not
        include the directory.
    """

    # check to see if directory exists; if it doesn't, create it.
    if not os.path.isdir(direc):
        os.makedirs(direc)

    filename = os.path.join(direc, filename)

    # check to see if file exists; if it doesn't, create it.
    if not os.path.exists(filename):
        open(filename, 'a').close()

    # write time to the file
    F = open(filename, 'a')
    F.write('%f\n' % t_total)
    F.close()


def reshape_soln_x(ug, nx, ny, p, px, py):
    """
    Reshape the solution if it is parallelized only in x. The solution must be
    of the form

        [ u0_0, u0_1, ..., u1_0, ..., ]

    where ui_j = row j of solution for process i. Note that this is NOT an
    array of arrays, rather, it is a flattened solution at a given time step.
    The reshaped solution should look like

        [ u0_0, u1_0, ..., ]
        [ u0_1, u1_1, ..., ]
        [ ... , ... , ..., ]

    Parameters
    ----------
    ug : ndarray
        The flattened solution at one time step.
    nx : int
        The number of x points per process.
    ny : int
        The number of y points per process.
    p  : int
        The total number of processes.
    px : int
        The number of processes in the x direction.
    py : int
        The number of processes in the y direction.
    """

    # evenly split ug into a list of p parts
    soln = np.array_split(ug, p)
    # reshape each part
    soln = [a.reshape(ny, nx) for a in soln]
    return np.hstack(soln)


def reshape_soln_y(ug, nx, ny, p, px, py):
    # evenly split ug into a list of p parts
    soln = np.array_split(ug, p)
    # reshape each part
    soln = np.hstack([a.reshape(ny, nx) for a in soln])
    soln = np.vstack([arr.transpose() for arr in np.array_split(soln.transpose(), p)])
    return soln


def reshape_soln_xy(ug, nx, ny, p, px, py):
    # evenly split ug into a list of p parts
    soln = np.array_split(ug, p)
    # reshape each part
    soln = np.hstack([a.reshape(ny, nx) for a in soln])

    temp = py*[None]
    for i in xrange(py):
        temp[i] = soln[:, i*px*nx : (i+1)*px*nx]

    return np.vstack(temp)
