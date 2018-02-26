import numpy as np
FLOAT_PRECISION = 1e-7


def fit_linear(y, x, find_intercept=True):
    cov = np.cov(y, x)
    b = cov[1, 0] / cov[1, 1]
    if find_intercept:
        return b, y.mean() - b * x.mean()
    else:
        return b


class SegmentedRegression:

    def __init__(self, window=15, eps=None):
        # origin data
        self.x = None
        self.y = None

        # cumulative sum data
        self._n = None
        self._x = None
        self._y = None
        self._xy = None
        self._x2 = None
        self._y2 = None

        # result
        self.splits_ = None

        # model parameters
        self.eps = eps
        self.window = window

    def fit(self, x, y):
        assert len(x.shape) == 1, 'input data x expects to be 1d array'
        assert len(y.shape) == 1, 'input data y expects to be 1d array'
        assert x.shape[0] == y.shape[0], 'input data must have same length'

        # TODO sort by x
        self.x = x
        self.y = y

        self._n = np.arange(1, x.shape[0] + 1)
        self._x = np.cumsum(x)
        self._y = np.cumsum(y)
        self._xy = np.cumsum(x * y)
        self._x2 = np.cumsum(x * x)
        self._y2 = np.cumsum(y * y)

        if self.eps is None:
            self.eps = 3 * self._estimate_var(x, y)

        self.splits_ = self._find_splits(0, x.shape[0], self.window, self.eps)

    def _find_splits(self, n1, n2, window, eps):
        # TODO save linear regression parameters on each interval
        n, v, v_r = self._get_variance_slice(n1, n2)

        if (n - n1 <= window) or (n + window >= n2):
            return []
        else:
            if v < eps and v_r < eps:
                return [n, ]
            elif v >= eps and v_r < eps:
                return self._find_splits(n1, n - window, window, eps) + [n, ]
            elif v < eps and v_r >= eps:
                return [n, ] + self._find_splits(n + window, n2, window, eps)
            else:
                return self._find_splits(n1, n - window, window, eps) + [n, ] + \
                       self._find_splits(n + window, n2, window, eps)

    def _get_variance_slice(self, n1, n2):
        # TODO reduce number of divisions by n
        x, y, xy, x2, y2, n = self._x, self._y, self._xy, self._x2, self._y2, self._n

        n_ = n[n1: n2] - (n[n1] - 1)
        x_m = (x[n1: n2] - x[n1]) / n_
        y_m = (y[n1: n2] - y[n1]) / n_
        xy_m = (xy[n1: n2] - xy[n1]) / n_
        cov = xy_m - x_m * y_m

        x2_m = (x2[n1: n2] - x2[n1]) / n_
        # add FLOAT_PRECISION to avoid division by zero
        var_x = x2_m - x_m * x_m + FLOAT_PRECISION

        y2_m = (y2[n1: n2] - y2[n1]) / n_
        var_y = y2_m - y_m * y_m

        v = var_y - cov * cov / var_x

        n2 -= 1  # last element will be added in the end
        n_ = n[n2] - n[n1: n2]
        x_m = (x[n2] - x[n1: n2]) / n_
        y_m = (y[n2] - y[n1: n2]) / n_
        xy_m = (xy[n2] - xy[n1: n2]) / n_
        cov = xy_m - x_m * y_m

        x2_m = (x2[n2] - x2[n1: n2]) / n_
        var_x = x2_m - x_m * x_m + FLOAT_PRECISION

        y2_m = (y2[n2] - y2[n1: n2]) / n_
        var_y = y2_m - y_m * y_m

        v_r = np.zeros_like(v)
        v_r[1:] = var_y - cov * cov / var_x

        n_ = n[n2]
        x_m = x[n2] / n_
        y_m = y[n2] / n_
        xy_m = xy[n2] / n_
        cov = xy_m - x_m * y_m

        x2_m = x2[n2] / n_
        var_x = x2_m - x_m * x_m + FLOAT_PRECISION

        y2_m = y2[n2] / n_
        var_y = y2_m - y_m * y_m

        v_r[0] = var_y - cov * cov / var_x

        n_relative = np.argmin(v + v_r)
        return n1 + n_relative, v[n_relative], v_r[n_relative]

    @staticmethod
    def _estimate_var(x, y):
        len_x = len(x)
        n = min(1000, len_x // 10)

        def get_var(y, x):
            cov = np.cov(y, x)
            return cov[0, 0] - cov[1, 0] * cov[1, 0] / cov[1, 1]

        var = [
            get_var(y[i1:i2], x[i1:i2])
            for i1, i2 in zip(range(0, len_x, n), range(n, len_x + n, n))
        ]

        return max(np.median(var), 1e-3)

    @property
    def lines(self):
        return [
            fit_linear(self.y[x1:x2], self.x[x1:x2])
            for x1, x2 in zip([0, ] + self.splits_, self.splits_ + [self.y.shape[0]])
        ]

    def predict(self, x):
        # TODO
        pass
