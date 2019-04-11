import numpy as np
import matplotlib.pyplot as plt


def grid_gen(grid_size=256, u_per=0.5, v_per=0.25, per_size=20):

    grid_u = np.ones((grid_size, grid_size))
    grid_v = np.zeros((grid_size, grid_size))

    x_start, y_start = int(grid_size/2 - 10), int(grid_size/2 - 10)
    x_end, y_end = x_start + per_size, y_start + per_size

    for i in range(x_start, x_end):
        for j in range(y_start, y_end):

            grid_u[i, j] = u_per
            grid_v[i, j] = v_per

            perturb = [0.99, 1.01]

            grid_u[i, j] *= perturb[np.random.randint(0, 2)]
            grid_v[i, j] *= perturb[np.random.randint(0, 2)]

    return grid_u, grid_v


def double_derivative_2d(grid, h):

    double_der_y = (np.roll(grid, 1, axis=0)+np.roll(grid, -1, axis=0)-2*grid)/(h*h)
    double_der_x = (np.roll(grid, 1, axis=1)+np.roll(grid, -1, axis=1)-2*grid)/(h*h)

    return double_der_x+double_der_y


def finite_diff(grid_u, grid_v, h, timesteps, dt, f, k, D_u, D_v):
    fig, ax = plt.subplots()
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    im = ax.imshow(grid_u, interpolation='nearest')
    fig.colorbar(im)

    for _ in range(timesteps):
        u_xy = double_derivative_2d(grid_u, h)
        v_xy = double_derivative_2d(grid_v, h)

        du_dt = -grid_u*grid_v**2 + f*(1 - grid_u) + D_u*u_xy
        dv_dt = grid_u*grid_v**2 - (f+k)*grid_v + D_v*v_xy
        grid_u += dt*du_dt
        grid_v += dt*dv_dt

    im.set_data(grid_u)
    fig.savefig('res_{}_{}.png'.format(f, k))


if __name__ == "__main__":
    grid_u, grid_v = grid_gen()
    finite_diff(grid_u, grid_v, 0.01, 50000, 1, 0.03, 0.062, 0.00002, 0.00001)
