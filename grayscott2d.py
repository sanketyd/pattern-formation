import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm


def grid_gen(grid_size=256, u_per=0.5, v_per=0.25, per_size=20, per_type='default', **kwargs):

    grid_u = np.ones((grid_size, grid_size))
    grid_v = np.zeros((grid_size, grid_size))

    x_start, y_start = int(grid_size/2 - per_size/2), int(grid_size/2 - per_size/2)
    x_end, y_end = x_start + per_size, y_start + per_size

    for i in range(x_start, x_end):
        for j in range(y_start, y_end):

            grid_u[i, j] = u_per
            grid_v[i, j] = v_per

            perturb = [0.99, 1.01]

            grid_u[i, j] *= perturb[np.random.randint(0, 2)]
            grid_v[i, j] *= perturb[np.random.randint(0, 2)]

    if per_type == 'overlap':
        x_start, y_start = int((x_start+x_end)/2), int((y_start+y_end)/2)
        x_end, y_end = int(x_start + per_size/2), int(y_start + per_size/2)

        for i in range(x_start, x_end):
            for j in range(y_start, y_end):

                grid_u[i, j] = kwargs['u_per_new']
                grid_v[i, j] = kwargs['v_per_new']

                perturb = [0.99, 1.01]

                grid_u[i, j] *= perturb[np.random.randint(0, 2)]
                grid_v[i, j] *= perturb[np.random.randint(0, 2)]

    return grid_u, grid_v


def double_derivative_2d(grid, h):

    double_der_y = (np.roll(grid, 1, axis=0)+np.roll(grid, -1, axis=0)-2*grid)/(h*h)
    double_der_x = (np.roll(grid, 1, axis=1)+np.roll(grid, -1, axis=1)-2*grid)/(h*h)

    return double_der_x+double_der_y


def finite_diff(grid_u, grid_v, h, timesteps, dt, f, k, D_u, D_v, frame_rate=100):
    fig, ax = plt.subplots()
    # ax.axes.get_yaxis().set_visible(False)
    # ax.axes.get_xaxis().set_visible(False)
    im = ax.imshow(grid_u, interpolation='nearest')
    fig.colorbar(im)

    counter = 1

    for t in tqdm(range(timesteps)):
        u_xy = double_derivative_2d(grid_u, h)
        v_xy = double_derivative_2d(grid_v, h)

        du_dt = -grid_u*grid_v**2 + f*(1 - grid_u) + D_u*u_xy
        dv_dt = grid_u*grid_v**2 - (f+k)*grid_v + D_v*v_xy
        grid_u += dt*du_dt
        grid_v += dt*dv_dt

        if t%frame_rate == 0:
            im.set_data(grid_u)
            fig.savefig('temp/_temp%06d.png'%counter)
            counter += 1

    im.set_data(grid_u)
    plt.title(r'f={}, k={}, $\Delta$t={}, $\Delta$x={}, $D_u$={}, $D_v$={}'.format(f, k, dt, h, D_u, D_v))
    fig.savefig('res_{}_{}.png'.format(f, k))


def make_animation(f, k):
    movie_name = 'mov_{}_{}.mpg'.format(f, k)
    os.system('rm {}'.format(movie_name))
    os.system('ffmpeg -r 25 -i temp/_temp%06d.png -b:v 1800 {}'.format(movie_name))
    os.system('rm temp/*')


if __name__ == "__main__":
    kwargs = {'u_per_new': 0.2, 'v_per_new': 0.6}
    grid_u, grid_v = grid_gen(u_per=0.6, v_per=0.6, per_size=10, per_type='overlap', **kwargs)
    # grid_u, grid_v = grid_gen()
    f, k = 0.062, 0.065
    finite_diff(grid_u, grid_v, 0.01, 100000, 0.5, f, k, 0.00002, 0.00001)
    make_animation(f, k)
