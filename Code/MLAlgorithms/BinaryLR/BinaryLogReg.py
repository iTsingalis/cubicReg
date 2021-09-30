import random
import numpy as np
from scipy.sparse import diags
from Utils.timer import Timer
from scipy.special import expit
from scipy.optimize import minimize
from Code.Optimizers.CubicReg import CubicReg
from sklearn.utils.validation import check_is_fitted


class BinaryLogReg:

    def __init__(self, atol=1e-4, reltol=1e-8, maxit=1000,
                 alpha=0.1, intersection=False, opt_method=None,
                 verbose=True, **kwargs):
        self.maxit = maxit
        self.reltol = reltol
        self.atol = atol
        self.alpha = alpha
        self.timer = Timer()
        self.verbose = verbose
        self.opt_method = opt_method
        self.intersection = intersection
        self.kwargs = kwargs

    @staticmethod
    def safe_log10(x, eps=1e-10):
        result = np.where(x > eps, x, -10)
        np.log10(result, out=result, where=result > 0)
        return result

    @staticmethod
    def sigmoid(w, X):
        a = X.dot(w)
        o = expit(a)
        return o

    @staticmethod
    def d_sigmoid(w, X):
        a = X.dot(w)
        o = expit(a)
        do = o * (1 - o)
        return do

    def cost(self, w):
        o = self.sigmoid(w, self.X)
        c = -(np.vdot(self.y, self.safe_log10(o)) + np.vdot(1 - self.y, self.safe_log10(1 - o))) / float(self.n_samples)
        return c

    def gradient(self, w):
        o = self.sigmoid(w, self.X)
        non_cvx_term = 2 * w[:, np.newaxis] / (w[:, np.newaxis] ** 2 + 1) ** 2
        grad = -self.X.T.dot(self.y - np.expand_dims(o, axis=1)) / float(self.n_samples) + self.alpha * non_cvx_term
        return np.squeeze(grad)

    def hessian(self, w):
        do = self.d_sigmoid(w, self.X)
        D = diags(do)
        non_cvx_term = 2 * np.eye(self.n_features) - 6 * np.diag(w[:, np.newaxis] ** 2) / (
                    1 + w[:, np.newaxis] ** 2) ** 3
        hs = self.X.T @ D @ self.X / self.n_samples + self.alpha * non_cvx_term
        return hs

    def grad_hess(self, w):
        # gradient AND hessian of the logistic
        grad = self.gradient(w, self.X, self.y, self.n_samples)
        Hs = self.hessian(w, self.X, self.n_samples)

        return grad, Hs.reshape(self.n_samples)

    def fit(self, X, y=None):
        self.timer.start()

        self.n_samples, self.n_features = X.shape

        y = np.reshape(y, (self.n_samples, 1))

        self.X, self.y = X, y

        random.seed(9)
        w_0 = np.random.uniform(-.1, .1, self.n_features)
        self.method_precision = []
        self.timings = []
        if self.opt_method == 'L-BFGS-B':
            from datetime import datetime

            start = datetime.now()

            def callback(w_0):
                prec = np.linalg.norm(self.gradient(w_0))
                self.method_precision.append(prec)
                self.timings.append((datetime.now() - start).total_seconds())

            callback(w_0)
            options = {'disp': self.verbose, 'maxiter': self.maxit}
            f_min = minimize(fun=self.cost, x0=w_0,
                             # args=(X, y, n_samples),
                             callback=callback,
                             method=self.opt_method,
                             jac=self.gradient,
                             hess=self.hessian,
                             tol=self.reltol,
                             options=options)
            self.coef_ = f_min.x[:, None]
        else:

            from datetime import datetime

            start = datetime.now()

            def callback(w_0):
                prec = np.linalg.norm(self.gradient(w_0))
                self.method_precision.append(prec)
                self.timings.append((datetime.now() - start).total_seconds())

            callback(w_0)

            cr = CubicReg(w_0, fun=self.cost,
                          grad=self.gradient,
                          hess=self.hessian,
                          callback=callback,
                          M=self.kwargs['kwargs']['M'],
                          L0=self.kwargs['kwargs']['L0'],
                          L=self.kwargs['kwargs']['L'],
                          kappa_easy=self.kwargs['kwargs']['kappa_easy'],
                          max_iter=self.kwargs['kwargs']['max_iter'],
                          max_sub_iter=self.kwargs['kwargs']['max_sub_iter'],
                          tol=self.kwargs['kwargs']['tol'],
                          stop_criteria=self.kwargs['kwargs']['stop_criteria'])
            x_opt, intermediate_points, n_iter = cr.fit()

            self.coef_ = x_opt

        self.timer.stop()

        return self

    def predict_proba(self, X):
        check_is_fitted(self, msg='not fitted.')

        sigma = self.sigmoid(self.coef_, X)

        return sigma

    def predict(self, X):
        check_is_fitted(self, msg='not fitted.')

        sigma = self.predict_proba(X)
        y_pred = [1 if x >= 0.5 else 0 for x in sigma]

        # y_pred = int(np.argmax(sigma, axis=1))
        return y_pred
