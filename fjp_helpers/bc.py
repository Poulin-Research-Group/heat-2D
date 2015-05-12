def set_periodic_BC(u):
    u[ 0, :] = u[-2, :]  # first row
    u[-1, :] = u[ 1, :]  # last row
    u[:,  0] = u[:, -2]  # first col
    u[:, -1] = u[:,  1]  # last col
    return u


def set_periodic_BC_x(u):
    # set X boundary conditions (cols)
    u[:,  0] = u[:, -2]  # first col
    u[:, -1] = u[:,  1]  # last col
    return u


def set_periodic_BC_y(u):
    # set periodic Y boundary conditions (rows)
    u[ 0, :] = u[-2, :]   # first row
    u[-1, :] = u[ 1, :]   # last row
    return u
