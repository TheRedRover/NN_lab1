import numpy as np

def weak_wolfe(network, network_loss_func, direction, grad):
    d = direction
    g_Sk = grad
    closure = network_loss_func
    F_k = closure()
    gtd = g_Sk.dot(d)

    closure_eval = 0

    max_ls = 10
    c1 = 1e-4
    c2 = 0.9

    # initialize counters
    ls_step = 0
    grad_eval = 0  # tracks gradient evaluations
    t_prev = 0  # old steplength

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
    print(
        '==================================== Begin Wolfe line search ====================================')
    print('F(x): %.8e  g*d: %.8e' % (F_k, gtd))

    # check if search direction is descent direction
    if gtd >= 0:
        desc_dir = False
        print('Not a descent direction!')
    else:
        desc_dir = True

    # update and evaluate at new point
    self._add_update(t, d)
    F_new = closure()
    closure_eval += 1

    # main loop
    while True:

        # check if maximum number of line search steps have been reached
        if ls_step >= max_ls:
            self._add_update(-t, d)

            t = 0
            F_new = closure()
            F_new.backward()
            g_new = self._gather_flat_grad()
            closure_eval += 1
            grad_eval += 1
            fail = True
            break

        # print info if debugging
        print('LS Step: %d  t: %.8e  alpha: %.8e  beta: %.8e'
              % (ls_step, t, alpha, beta))
        print('Armijo:  F(x+td): %.8e  F-c1*t*g*d: %.8e  F(x): %.8e'
              % (F_new, F_k + c1 * t * gtd, F_k))

        # check Armijo condition
        if F_new > F_k + c1 * t * gtd:

            # set upper bound
            beta = t
            t_prev = t

            # update interpolation quantities
            if interpolate:
                F_b = F_new
                if torch.cuda.is_available():
                    g_b = torch.tensor(np.nan, dtype=dtype).cuda()
                else:
                    g_b = torch.tensor(np.nan, dtype=dtype)

        else:

            # compute gradient
            F_new.backward()
            g_new = self._gather_flat_grad()
            grad_eval += 1
            gtd_new = g_new.dot(d)

            # print info if debugging
            if ls_debug:
                print('Wolfe: g(x+td)*d: %.8e  c2*g*d: %.8e  gtd: %.8e'
                      % (gtd_new, c2 * gtd, gtd))

            # check curvature condition
            if gtd_new < c2 * gtd:

                # set lower bound
                alpha = t
                t_prev = t

                # update interpolation quantities
                if interpolate:
                    F_a = F_new
                    g_a = gtd_new

            else:
                break

        # compute new steplength

        # if first step or not interpolating, then bisect or multiply by factor
        if not interpolate or not is_legal(F_b):
            if beta == float('Inf'):
                t = eta * t
            else:
                t = (alpha + beta) / 2.0

        # otherwise interpolate between a and b
        else:
            t = polyinterp(np.array([[alpha, F_a.item(), g_a.item()], [beta, F_b.item(), g_b.item()]]))

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
        self._add_update(t - t_prev, d)

        # evaluate closure
        F_new = closure()
        closure_eval += 1
        ls_step += 1

    # store Bs
    if Bs is None:
        Bs = (g_Sk.mul(-t)).clone()
    else:
        Bs.copy_(g_Sk.mul(-t))

    # print final steplength
    print('Final Steplength:', t)
    print('===================================== End Wolfe line search =====================================')
