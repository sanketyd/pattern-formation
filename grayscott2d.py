import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm


def grid_gen(grid_size=256, u_per=0.5, v_per=0.25, per_size=20, num_per=1, per_type='default', **kwargs):
    """
    This method generates a grid and initialize it based on user input.
    :param grid_size: Size of square grid.
    :param u_per: Concentration of u in perturbed area.
    :param v_per: Concentration of v in perturbed area.
    :param per_size: Size of perturbations.
    :param num_per: Number of perturbations.
    :param per_type: Number of perturbations.
    :param kwargs: Other arguments
    :return: grid_u, grid_v
    """

    grid_u = np.ones((grid_size, grid_size))
    grid_v = np.zeros((grid_size, grid_size))

    if num_per == 1:
        per_gen(grid_u, grid_v, grid_size/2, grid_size/2, u_per, v_per, per_size, per_type, **kwargs)
    else:
        for _ in range(num_per):
            x_init, y_init = np.random.randint(20, grid_size-20), np.random.randint(20, grid_size-20)
            per_gen(grid_u, grid_v, x_init, y_init, u_per, v_per, per_size, per_type, **kwargs)

    return grid_u, grid_v


def per_gen(grid_u, grid_v, x_init, y_init, u_per, v_per, per_size, per_type, **kwargs):
    """
    This method is helper for gen_grid to generate perturbations.
    """

    x_start, y_start = int(x_init - per_size/2), int(y_init - per_size/2)
    x_end, y_end = x_start + per_size, y_start + per_size

    for i in range(x_start, x_end):
        for j in range(y_start, y_end):

            grid_u[i, j] = u_per
            grid_v[i, j] = v_per

            # following 3 lines make +-1% change randomly
            perturb = [0.99, 1.01]

            grid_u[i, j] *= perturb[np.random.randint(0, 2)]
            grid_v[i, j] *= perturb[np.random.randint(0, 2)]

    if per_type == 'overlap':
        if np.random.randint(0, 2):
            x_start, y_start = int((x_start+x_end)/2), y_start
        else:
            x_start, y_start = int((x_start+x_end)/2), int((y_start+y_end)/2)
        x_end, y_end = int(x_start + per_size/2), int(y_start + per_size/2)

        # if multiple concentrations are supplied select one of them randomly
        rand_index = None
        if type(kwargs['u_per_new']) is list:
            rand_index = np.random.randint(0, len(kwargs['u_per_new']))

        for i in range(x_start, x_end):
            for j in range(y_start, y_end):

                if type(kwargs['u_per_new']) is list:
                    grid_u[i, j] = kwargs['u_per_new'][rand_index]
                else:
                    grid_u[i, j] = kwargs['u_per_new']
                grid_v[i, j] = kwargs['v_per_new']

                perturb = [0.99, 1.01]

                grid_u[i, j] *= perturb[np.random.randint(0, 2)]
                grid_v[i, j] *= perturb[np.random.randint(0, 2)]


def double_derivative_2d(grid, h):
    """
    This method returns laplacian of grid with Periodic boundary conditions.
    :param grid: variable
    :param h: step size
    :return: laplacian of grid
    """

    double_der_y = (np.roll(grid, 1, axis=0)+np.roll(grid, -1, axis=0)-2*grid)/(h*h)
    double_der_x = (np.roll(grid, 1, axis=1)+np.roll(grid, -1, axis=1)-2*grid)/(h*h)

    return double_der_x+double_der_y


def finite_diff(grid_u, grid_v, h, timesteps, dt, f, k, D_u, D_v, method='euler', frame_rate=100):
    """
    This method implements finite difference method.
    :param grid_u: u of differential equation
    :param grid_v: v of differential equation
    :param timesteps: Number of timesteps
    :param dt: Timestep
    :param f: feed rate
    :param k: death rate
    :param D_u: Diffusion rate of u
    :param D_v: Diffusion rate of v
    :param method: Integration method can be 'euler', 'rk2' or 'rk4'
    :param frame_rate: After how many iterations we capture a frame.
    """

    fig, ax = plt.subplots()
    im = ax.imshow(grid_u, interpolation='nearest')
    fig.colorbar(im)

    counter = 1

    for t in tqdm(range(timesteps)):
        u_xy = double_derivative_2d(grid_u, h)
        v_xy = double_derivative_2d(grid_v, h)

        grid_u_v2 = grid_u*grid_v**2

        def du_dt(gu):
            return -grid_u_v2 + f*(1 - gu) + D_u*u_xy

        def dv_dt(gv):
            return grid_u_v2 - (f+k)*gv + D_v*v_xy
        if method == 'euler':
            gu, gv = grid_u, grid_v
            grid_u += dt*du_dt(gu)
            grid_v += dt*dv_dt(gv)
        elif method == 'rk2':
            gu, gv = grid_u, grid_v
            k1 = dt*0.5*du_dt(gu)
            grid_u += dt*du_dt(gu+k1)
            k1 = dt*0.5*dv_dt(gv)
            grid_v += dt*dv_dt(gv+k1)
        elif method == 'rk4':
            gu, gv = grid_u, grid_v
            k1 = dt*du_dt(gu)
            k2 = dt*du_dt(gu + 0.5*k1)
            k3 = dt*du_dt(gu + 0.5*k2)
            k4 = dt*du_dt(gu + k3)
            grid_u += (k1 + 2*k2 + 2*k3 + k4)/6.0
            k1 = dt*dv_dt(gv)
            k2 = dt*dv_dt(gv + 0.5*k1)
            k3 = dt*dv_dt(gv + 0.5*k2)
            k4 = dt*dv_dt(gv + k3)
            grid_v += (k1 + 2*k2 + 2*k3 + k4)/6.0

        if t%frame_rate == 0:
            im.set_data(grid_u)
            plt.title(r'f={}, k={}, $\Delta$t={}, $\Delta$x={}, $D_u$={}, $D_v$={}'.format(f, k, dt, h, D_u, D_v))
            fig.savefig('temp/_temp%06d.jpg'%counter)
            counter += 1

    im.set_data(grid_u)
    plt.title(r'f={}, k={}, $\Delta$t={}, $\Delta$x={}, $D_u$={}, $D_v$={}'.format(f, k, dt, h, D_u, D_v))
    fig.savefig('res_{}_{}.png'.format(f, k))


def make_animation(f, k):
    """
    Make animation
    """
    movie_name = 'mov_{}_{}.mpg'.format(f, k)
    os.system('rm {}'.format(movie_name))
    os.system('ffmpeg -r 25 -i temp/_temp%06d.jpg -vb 20M {}'.format(movie_name))
    os.system('rm temp/*')


if __name__ == "__main__":
    # inputs (f, k, initialization type)
    inp = [(0.062, 0.065, 'overlap'), (0.04, 0.06, 'default'), (0.035, 0.065, 'default'), (0.012, 0.05, 'default'), (0.025, 0.05, 'default')]
    for params in inp:
        if params[2] == 'overlap':
            kwargs = {'u_per_new': [0.2, 0.6], 'v_per_new': 0.6}
            grid_u, grid_v = grid_gen(u_per=0.6, v_per=0.6, per_size=10, num_per=15, per_type='overlap', **kwargs)
        else:
            grid_u, grid_v = grid_gen()
        finite_diff(grid_u, grid_v, 0.01, 100000, 0.5, params[0], params[1], 0.00002, 0.00001, 'rk4')
        make_animation(params[0], params[1])
