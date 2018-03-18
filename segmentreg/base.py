import numpy as np

FLOAT_PRECISION = 1e-14


class SegmentedRegression:
    """Segmented linear regression model.

    Parameters
    ----------
    min_segment_len : int, optional
        limit of distance in points of two sequential break points

    eps : float, optional
        threshold when stop further split search
    """

    def __init__(self, min_segment_len=15, eps=None):
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
        self.segments_ = None

        # model parameters
        self.eps = eps
        self.min_seg = int(min_segment_len)

    def fit(self, x, y, is_sorted=False):
        """Find segments and apply linear fit
        :param x : 1d array, Data
        :param y : 1d array, Target
        :param is_sorted : boolean, (default=False) Whether x is already sorted
        """
        assert len(x.shape) == 1, 'input data x expects to be 1d array'
        assert len(y.shape) == 1, 'input data y expects to be 1d array'
        assert x.shape[0] == y.shape[0], 'input data must have same length'

        if not is_sorted:
            idx = np.argsort(x)
            x, y = x[idx], y[idx]

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

        self.segments_ = self._find_segments(0, x.shape[0])

    def _find_segments(self, n1, n2):
        window, eps = self.min_seg, self.eps
        n, v, v_r = self._get_variance_slice(n1, n2)

        if (n - n1 <= window) or (n + window >= n2):
            return [self._get_segment_info(n1, n2), ]
        else:
            if v < eps and v_r < eps:
                return [self._get_segment_info(n1, n), self._get_segment_info(n, n2)]
            elif v >= eps and v_r < eps:
                return self._find_segments(n1, n) + [self._get_segment_info(n, n2), ]
            elif v < eps and v_r >= eps:
                return [self._get_segment_info(n1, n), ] + self._find_segments(n, n2)
            else:
                return self._find_segments(n1, n) + self._find_segments(n, n2)

    def _get_segment_info(self, n1, n2):
        # because of precision issue we can not use cumsum values to estimate slope and intercept
        x, y = self.x[n1: n2], self.y[n1: n2]
        cov = np.cov(x, y)
        if cov[0, 0] < FLOAT_PRECISION:
            k = np.inf
            b = np.nan
        else:
            k = cov[0, 1] / cov[0, 0]
            b = y.mean() - k * x.mean()
        return x[0], k, b

    def _get_variance_slice(self, n1, n2):
        # TODO reduce number of divisions by n
        x, y, xy, x2, y2, n = self._x, self._y, self._xy, self._x2, self._y2, self._n

        n_ = n[n1: n2] - (n[n1] - 1)
        x_m = (x[n1: n2] - x[n1]) / n_
        y_m = (y[n1: n2] - y[n1]) / n_
        xy_m = (xy[n1: n2] - xy[n1]) / n_
        cov = xy_m - x_m * y_m

        x2_m = (x2[n1: n2] - x2[n1]) / n_
        var_x = x2_m - x_m * x_m
        var_x[var_x < FLOAT_PRECISION] = np.nan  # to avoid division by zero

        y2_m = (y2[n1: n2] - y2[n1]) / n_
        var_y = y2_m - y_m * y_m

        v = var_y - cov * cov / var_x
        v[0] = np.nan  # one point estimation is rubbish

        n2 -= 1  # last element will be added in the end
        n_ = n[n2] - n[n1: n2]
        x_m = (x[n2] - x[n1: n2]) / n_
        y_m = (y[n2] - y[n1: n2]) / n_
        xy_m = (xy[n2] - xy[n1: n2]) / n_
        cov = xy_m - x_m * y_m

        x2_m = (x2[n2] - x2[n1: n2]) / n_
        var_x = x2_m - x_m * x_m
        var_x[var_x < FLOAT_PRECISION] = np.nan  # to avoid division by zero

        y2_m = (y2[n2] - y2[n1: n2]) / n_
        var_y = y2_m - y_m * y_m

        v_r = np.zeros_like(v)
        v_r[1:] = var_y - cov * cov / var_x

        try:
            n_relative = np.nanargmin(v + v_r)
            return n1 + n_relative, v[n_relative], v_r[n_relative]
        except:
            # if all values is nan
            return n1, 0, 0 

    @staticmethod
    def _estimate_var(x, y):
        len_x = len(x)
        n = min(1000, len_x // 10)

        def get_var(x, y):
            cov = np.cov(x, y)
            return cov[1, 1] - (cov[1, 0] * cov[1, 0] / cov[0, 0] if cov[0, 0] > FLOAT_PRECISION else 0.)

        var = [
            get_var(x[i1:i2], y[i1:i2])
            for i1, i2 in zip(range(0, len_x, n), range(n, len_x, n))
        ]

        return np.median(var)

    def predict(self, x):
        segments = self.segments_
        if len(segments) == 1:
            _, k, b = segments[0]
            return k * x + b
        else:
            condlist = [x < segments[1][0], ] + [
                x >= segment[0]
                for segment in segments[1:]
            ]
            funclist = [
                lambda x, k=k, b=b: k*x + b
                for _, k, b in segments
            ]
            return np.piecewise(x, condlist, funclist)
