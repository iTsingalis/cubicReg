import numpy as np
import scipy.linalg


class CubicReg:
    """
      Minimize Newton's cubic regularization method using Algorithm 7.3.6 in [1].
      References
      ----------
      [1] Conn, A. R., Gould, N. I., & Toint, P. L. (2000). Trust region methods. Society for Industrial and Applied Mathematics.
      [2] Nesterov, Y., & Polyak, B. T. (2006). Cubic regularization of Newton method and its global performance. Mathematical Programming, 108(1), 177-205.
    """

    def __init__(self, x0, fun, grad, hess, M=None,
                 L0=None, L=None, kappa_easy=0.1,
                 max_iter=10000,
                 max_sub_iter=100,
                 max_line_search_iter=1000,
                 tol=1e-4,
                 callback=None,
                 epsilon=2 * np.sqrt(np.finfo(float).eps),
                 stop_criteria='nesterov'):

        self.M = M
        self.L = L
        self.L0 = L0
        self.x0 = x0
        self.fun = fun
        self.tol = tol
        self.grad = grad
        self.hess = hess
        self.epsilon = epsilon
        self.callback = callback
        self.max_iter = max_iter
        self.kappa_easy = kappa_easy
        self.max_sub_iter = max_sub_iter
        self.stop_criteria = stop_criteria
        self.max_line_search_iter = max_line_search_iter

        self.grad_x = self.grad(self.x0)
        self.hess_x = self.hess(self.x0)

        self.H_lambda = lambda lambda_k: self.hess_x + lambda_k * np.identity(np.size(self.hess_x, 0))
        self.lambda_const = lambda lambda_k: (1 + lambda_k) * np.sqrt(np.finfo(float).eps)

        if L0 is None:
            self.L0 = np.linalg.norm(self.hess(self.x0) - self.hess(self.x0 + np.ones_like(self.x0)),
                                     ord=2) / np.linalg.norm(np.ones_like(self.x0)) + self.epsilon

        if not self.stop_criteria.lower() in ['nesterov', 'gradient']:
            raise ValueError('stopping criteria should be "nesterov" of "gradient".')
        if self.stop_criteria.lower() == 'nesterov' and self.L is None:
            raise ValueError("Parameter L must be specified when Nesterov's stopping criterion is used.")

    def _lambda_one_plus(self):

        lambda_n, u_n = scipy.linalg.eigh(self.hess_x, eigvals=(0, 0))
        return max(-lambda_n[0], 0), lambda_n, np.squeeze(u_n)

    def _check_convergence(self, lambda_n, M):

        if self.stop_criteria.lower() == 'gradient':
            if np.linalg.norm(self.grad_x) <= self.tol:
                return True
            else:
                return False
        elif self.stop_criteria.lower() == 'nesterov':
            if max(np.sqrt(2 / (self.L + M) * np.linalg.norm(self.grad_x)), -2 / (2 + M) * lambda_n) <= self.tol:
                return True
            else:
                return False

    def fit(self):

        self._n_iter = 1
        line_search_iter = 0
        x_new = self.x0

        intermediate_points = [list(self.x0) + [0]]

        # Algorithm (3.3) in [2]
        while self._n_iter < self.max_iter:
            if self.callback is not None:
                self.callback(np.copy(x_new))
            x_old = x_new.copy()
            lambda_k, lambda_n, u_n = self._lambda_one_plus()
            if self.M:  # Exact M
                mk = self.M
                s = self.argmin_TM(mk=self.M, lambda_k=lambda_k,
                                              lambda_n=lambda_n, u_n=u_n)
                x_new = s + x_old
            else:  # find M_k using line search
                mk = self.M if self.M else self.L0
                x_new, mk, line_search_iter = self._line_search(x_old=x_old, mk=mk,
                                                                           lambda_k=lambda_k,
                                                                           lambda_n=lambda_n,
                                                                           u_n=u_n)

            self.grad_x = self.grad(x_new)
            self.hess_x = self.hess(x_new)

            # Section 3 in [2], see convergence criteria.
            if self._check_convergence(lambda_n, mk):
                break

            intermediate_points += [(list(x_new) + [self._n_iter])]
            self._n_iter += 1
            print('main_iter {} - lin_search_iter {}/{} - norm_grad {} - mk {}'
                  .format(self._n_iter, line_search_iter, 2 * self._n_iter + np.log2(mk / self.L0),
                          np.linalg.norm(self.grad_x), mk))
        return x_new, intermediate_points, self._n_iter

    def _line_search(self, x_old, mk, lambda_k, lambda_n, u_n):

        line_search_iter = 1
        f_xold = self.fun(x_old)
        while True:
            mk *= 2
            s = self.argmin_TM(mk=mk, lambda_k=lambda_k,
                                          lambda_n=lambda_n, u_n=u_n)
            x_new = s + x_old

            if self.fun(x_new) - f_xold < 1e-6:
                break

            if line_search_iter > 2 * self._n_iter + np.log2(mk / self.L0):  # self.max_line_search_iter:
                return x_new, mk / 2, line_search_iter

            line_search_iter += 1

        mk = max(0.5 * mk, self.L0)
        return x_new, mk, line_search_iter

    def argmin_TM(self, mk, lambda_k, lambda_n, u_n):

        """
            Solve the cubic regularization subproblem. See algorithm 7.3.6 in Conn et al. (2000).
        """

        # STEP 1, Algorithm 7.3.6
        # self.lambda_k, self.lambda_n, self.u_n = self._lambda_one_plus()
        lambda_const = self.lambda_const(lambda_k)
        if lambda_k == 0:
            lambda_k = 0
        else:
            lambda_k = lambda_k + lambda_const

        s, L = self._compute_s(lambda_k, lambda_const)  # Step 2. Algorithm 7.3.6

        r = 2 * lambda_k / mk
        if np.linalg.norm(s) <= r:  # r == \Delta for Algorithm 7.3.6 in [1]
            if lambda_k == 0 or np.linalg.norm(s) == r:  # Step 3a, Algorithm 7.3.6 in [1]
                return s
            else:
                Lambda, U = np.linalg.eigh(self.H_lambda(lambda_k))
                s_cri = - U.T.dot(np.linalg.pinv(np.diag(Lambda - lambda_n))).dot(U).dot(self.grad_x)
                alpha = min(np.roots([np.dot(u_n, u_n), 2 * np.dot(u_n, s_cri),
                                      np.dot(s_cri, s_cri) - 4 * lambda_k ** 2 / mk ** 2]))
                s = s_cri + alpha * u_n

                return s

        if lambda_k == 0:
            lambda_k += lambda_const

        sub_iter = 1
        lambda_const = self.lambda_const(lambda_k)
        while True:
            lambda_k = self._lambda_next(lambda_k, s, L, mk)
            # Step 2. Algorithm 7.3.6 in [1]
            s, L = self._compute_s(lambda_k, lambda_const)
            sub_iter += 1
            if sub_iter > self.max_sub_iter:
                print(RuntimeWarning('Maximum number of sub iterations exceeded in Steps 4.-5. for Algorithm 7.3.6 in [1]'))
                break
            if self._converged(s, lambda_k, mk):
                break

        return s

    def _compute_s(self, lambda_k, lambda_const):
        # STEP 2. Algorithm 7.3.6 in [1]
        try:
            L = np.linalg.cholesky(self.H_lambda(lambda_k)).T
        except np.linalg.LinAlgError:
            lambda_const *= 2
            # RecursionError: maximum recursion depth exceeded while calling a Python object
            s, L = self._compute_s(lambda_k + lambda_const, lambda_const)

        s = scipy.linalg.cho_solve((L, False), -self.grad_x)

        return s, L

    def _lambda_next(self, lambda_k, s, L, mk):

        w = scipy.linalg.solve_triangular(L.T, s, lower=True)
        norm_s = np.linalg.norm(s)

        phi = 1 / norm_s - mk / (2 * lambda_k)

        phi_prime = np.linalg.norm(w) ** 2 / (norm_s ** 3) + mk / (2 * lambda_k ** 2)
        return lambda_k - phi / phi_prime

    def _converged(self, s, lambda_k, mk):

        r = 2 * lambda_k / mk
        if abs(np.linalg.norm(s) - r) <= self.kappa_easy * r:
            return True
        else:
            return False
