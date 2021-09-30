import numpy as np


class GradDesc:

    def __init__(self, x0, fun, grad, lr=1e-3, max_iter=1000, tol=1e-3, conv_criterion='grad'):
        self.x0 = x0
        self.lr = lr
        self.fun = fun
        self.tol = tol
        self.grad = grad
        self.max_iter = max_iter
        self.conv_criterion = conv_criterion

        self.grad_x = self.grad(self.x0)


        if self.conv_criterion == 'grad':
            self.conv_fun = lambda grad_x: np.linalg.norm(grad_x)
        elif self.conv_criterion == 'diff_points':
            self.conv_fun = lambda x_old, x_new: np.linalg.norm(x_old - x_new)

    def _check_convergence(self, x_new, x_old, n_iter):

        if self.conv_criterion == 'grad':
            conv_fun = self.conv_fun(self.grad_x)
            if conv_fun <= self.tol:
                return True
        elif self.conv_criterion == 'diff_points':
            conv_fun = self.conv_fun(x_new, x_old)
            if conv_fun <= self.tol:
                return True

        if n_iter > self.max_iter:
            return True

        print('n_iter {} - {} - {}'.format(n_iter, self.conv_criterion, conv_fun))

        return False

    def fit(self):

        n_iter = 1
        x_new = self.x0
        intermediate_points = [self.x0 + [0]]

        while True:
            x_old = x_new.copy()

            x_new = self.partial_fit(x_old)

            self.grad_x = self.grad(x_new)

            intermediate_points += [(list(x_new) + [n_iter])]
            n_iter += 1

            if self._check_convergence(x_new, x_old, n_iter):
                break

        return x_new, intermediate_points, n_iter

    def partial_fit(self, x_old):

        x_new = x_old - self.lr * self.grad_x

        return x_new


