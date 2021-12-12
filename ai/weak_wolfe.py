import numpy as np

from ai.layers.layer_dense import LayerDense


def weak_wolfe(layer: LayerDense, network_loss_func, direction, grad, lr):
    def add_update(step, direction_flat, layer):
        layer.weights += (step * direction_flat).reshape(layer.weights.shape)

    debug = False
    d = direction
    g_Sk = grad
    closure = lambda: network_loss_func()[0]
    F_k = closure()
    gtd = g_Sk.dot(d)

    closure_eval = 0

    max_ls = 10
    c1 = 1e-4
    c2 = 0.9
    eta = 2

    # initialize counters
    ls_step = 0
    grad_eval = 0  # tracks gradient evaluations
    t_prev = 0  # old steplength

    t = lr

    # initialize bracketing variables and flag
    alpha = 0
    beta = float('Inf')
    fail = False

    # initialize values for line search
    F_a = F_k
    g_a = gtd

    F_b = np.nan
    g_b = np.nan

    # begin print for debug mode
    if debug:
        print(
            '==================================== Begin Wolfe line search ====================================')
        print(f'F(x): {F_k}  g*d: {gtd}')

    # check if search direction is descent direction
    if gtd >= 0:
        desc_dir = False
        print('Not a descent direction!')
    else:
        desc_dir = True

    weights_before_updates = layer.weights.copy()

    # update and evaluate at new point
    add_update(t, d, layer)
    F_new = closure()
    closure_eval += 1
    grad_eval += 1
    fail = False

    # main loop
    while True:

        # check if maximum number of line search steps have been reached
        if ls_step >= max_ls:
            add_update(-t, d, layer)

            t = 0
            F_new = closure()
            g_new = layer.dweights.flatten()
            closure_eval += 1
            grad_eval += 1
            fail = True
            if debug:
                print("FAIL")
            break

        # print info if debugging
        if debug:
            print(f'LS Step: {ls_step}  t: {t}  alpha: {alpha}  beta: {beta}')
            print(f'Armijo:  F(x+td): {F_new}  F-c1*t*g*d: {F_k + c1 * t * gtd}  F(x): {F_k}')

        # check Armijo condition
        if F_new > F_k + c1 * t * gtd:
            # set upper bound
            beta = t
            t_prev = t

            # update interpolation quantities
            F_b = F_new
            g_b = np.nan

        else:
            # compute gradient
            g_new = layer.dweights.flatten()
            closure_eval += 1
            grad_eval += 1
            gtd_new = g_new.dot(d)

            # print info if debugging
            if debug:
                print(f'Wolfe: g(x+td)*d: {gtd_new}  c2*g*d: {c2 * gtd}  gtd: {gtd}')

            # check curvature condition
            if gtd_new < c2 * gtd:

                # set lower bound
                alpha = t
                t_prev = t

                # update interpolation quantities
                F_a = F_new
                g_a = gtd_new

            else:
                break

        # compute new steplength

        # if first step or not interpolating, then bisect or multiply by factor
        if not is_legal(F_b):
            if beta == float('Inf'):
                t = eta * t
            else:
                t = (alpha + beta) / 2.0

        # otherwise interpolate between a and b
        else:
            t = polyinterp(np.array([[alpha, F_a, g_a], [beta, F_b, g_b]]))

            # if values are too extreme, adjust t
            if beta == float('Inf'):
                if t > 2 * eta * t_prev:
                    t = 2 * eta * t_prev
                elif t < eta * t_prev:
                    t = eta * t_prev
            else:
                if t < alpha + 0.2 * (beta - alpha):
                    t = alpha + 0.2 * (beta - alpha)
                elif t > (beta - alpha) / 2.0:
                    t = (beta - alpha) / 2.0

            # if we obtain nonsensical value from interpolation
            if t <= 0:
                t = (beta - alpha) / 2.0

        # update parameters
        add_update(t - t_prev, d, layer)
        # evaluate closure
        F_new = closure()
        closure_eval += 1
        ls_step += 1

    # print final steplength
    if debug:
        print('Final Steplength:', t)
        print('===================================== End Wolfe line search =====================================')
    layer.weights = weights_before_updates
    closure()

    return t, fail


def is_legal(v):
    """
    Checks that tensor is not NaN or Inf.

    Inputs:
        v (tensor): tensor to be checked

    """
    legal = not np.isnan(v).any() and not np.isinf(v)

    return legal


def polyinterp(points, x_min_bound=None, x_max_bound=None, plot=False):
    """
    Gives the minimizer and minimum of the interpolating polynomial over given points
    based on function and derivative information. Defaults to bisection if no critical
    points are valid.

    Based on polyinterp.m Matlab function in minFunc by Mark Schmidt with some slight
    modifications.

    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 12/6/18.

    Inputs:
        points (nparray): two-dimensional array with each point of form [x f g]
        x_min_bound (float): minimum value that brackets minimum (default: minimum of points)
        x_max_bound (float): maximum value that brackets minimum (default: maximum of points)
        plot (bool): plot interpolating polynomial

    Outputs:
        x_sol (float): minimizer of interpolating polynomial
        F_min (float): minimum of interpolating polynomial

    Note:
      . Set f or g to np.nan if they are unknown

    """
    no_points = points.shape[0]
    order = np.sum(1 - np.isnan(points[:, 1:3]).astype('int')) - 1

    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])

    # compute bounds of interpolation area
    if x_min_bound is None:
        x_min_bound = x_min
    if x_max_bound is None:
        x_max_bound = x_max

    # explicit formula for quadratic interpolation
    if no_points == 2 and order == 2 and plot is False:
        # Solution to quadratic interpolation is given by:
        # a = -(f1 - f2 - g1(x1 - x2))/(x1 - x2)^2
        # x_min = x1 - g1/(2a)
        # if x1 = 0, then is given by:
        # x_min = - (g1*x2^2)/(2(f2 - f1 - g1*x2))

        if points[0, 0] == 0:
            x_sol = -points[0, 2] * points[1, 0] ** 2 / (
                    2 * (points[1, 1] - points[0, 1] - points[0, 2] * points[1, 0]))
        else:
            a = -(points[0, 1] - points[1, 1] - points[0, 2] * (points[0, 0] - points[1, 0])) / (
                    points[0, 0] - points[1, 0]) ** 2
            x_sol = points[0, 0] - points[0, 2] / (2 * a)

        x_sol = np.minimum(np.maximum(x_min_bound, x_sol), x_max_bound)

    # explicit formula for cubic interpolation
    elif no_points == 2 and order == 3 and plot is False:
        # Solution to cubic interpolation is given by:
        # d1 = g1 + g2 - 3((f1 - f2)/(x1 - x2))
        # d2 = sqrt(d1^2 - g1*g2)
        # x_min = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2))
        d1 = points[0, 2] + points[1, 2] - 3 * ((points[0, 1] - points[1, 1]) / (points[0, 0] - points[1, 0]))
        d2 = np.sqrt(d1 ** 2 - points[0, 2] * points[1, 2])
        if np.isreal(d2):
            x_sol = points[1, 0] - (points[1, 0] - points[0, 0]) * (
                    (points[1, 2] + d2 - d1) / (points[1, 2] - points[0, 2] + 2 * d2))
            x_sol = np.minimum(np.maximum(x_min_bound, x_sol), x_max_bound)
        else:
            x_sol = (x_max_bound + x_min_bound) / 2

    # solve linear system
    else:
        # define linear constraints
        A = np.zeros((0, order + 1))
        b = np.zeros((0, 1))

        # add linear constraints on function values
        for i in range(no_points):
            if not np.isnan(points[i, 1]):
                constraint = np.zeros((1, order + 1))
                for j in range(order, -1, -1):
                    constraint[0, order - j] = points[i, 0] ** j
                A = np.append(A, constraint, 0)
                b = np.append(b, points[i, 1])

        # add linear constraints on gradient values
        for i in range(no_points):
            if not np.isnan(points[i, 2]):
                constraint = np.zeros((1, order + 1))
                for j in range(order):
                    constraint[0, j] = (order - j) * points[i, 0] ** (order - j - 1)
                A = np.append(A, constraint, 0)
                b = np.append(b, points[i, 2])

        # check if system is solvable
        if A.shape[0] != A.shape[1] or np.linalg.matrix_rank(A) != A.shape[0]:
            x_sol = (x_min_bound + x_max_bound) / 2
            f_min = np.Inf
        else:
            # solve linear system for interpolating polynomial
            coeff = np.linalg.solve(A, b)

            # compute critical points
            dcoeff = np.zeros(order)
            for i in range(len(coeff) - 1):
                dcoeff[i] = coeff[i] * (order - i)

            crit_pts = np.array([x_min_bound, x_max_bound])
            crit_pts = np.append(crit_pts, points[:, 0])

            if not np.isinf(dcoeff).any():
                roots = np.roots(dcoeff)
                crit_pts = np.append(crit_pts, roots)

            # test critical points
            f_min = np.Inf
            x_sol = (x_min_bound + x_max_bound) / 2  # defaults to bisection
            for crit_pt in crit_pts:
                if np.isreal(crit_pt) and crit_pt >= x_min_bound and crit_pt <= x_max_bound:
                    F_cp = np.polyval(coeff, crit_pt)
                    if np.isreal(F_cp) and F_cp < f_min:
                        x_sol = np.real(crit_pt)
                        f_min = np.real(F_cp)

            # if (plot):
            #     plt.figure()
            #     x = np.arange(x_min_bound, x_max_bound, (x_max_bound - x_min_bound) / 10000)
            #     f = np.polyval(coeff, x)
            #     plt.plot(x, f)
            #     plt.plot(x_sol, f_min, 'x')

    return x_sol
