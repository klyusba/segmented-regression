import numpy as np
import bottleneck as bn

FLOAT_PRECISION = 1e-14


def move_var(x):
    return bn.move_var(x, len(x), 1)


def join_var(x, y):
    var_x, var_y = move_var(x), move_var(y)
    k = np.sqrt(var_y[-1] / var_x[-1])
    cov = (move_var(y + k * x) - var_y - k ** 2 * var_x) / (2 * k)

    i = np.where(var_x < FLOAT_PRECISION)[0].max() + 1
    v = var_y
    if i < len(x):
        v[i:] -= cov[i:] * cov[i:] / var_x[i:]
    return v


class SegmentedRegression:
    """Segmented linear regression model.

    Parameters
    ----------
    min_segment_len : int, optional
        limit of distance in points of two sequential break points

    eps : float, optional
        threshold when stop further split search

    Attributes
    ----------
    segments_ : list of tuples,
        segment info: x_start, slope, intercept

    Examples
    --------
    see examples folder

    """

    def __init__(self, min_segment_len=15, eps=None):
        # origin data
        self.x = None
        self.y = None

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

        if self.eps is None:
            self.eps = 3 * self._estimate_var(x, y)

        self.segments_ = self._find_segments(0, x.shape[0])

    def _find_segments(self, n1, n2):
        window, eps = self.min_seg, self.eps
        n, v, v_r = self._get_variance_slice(n1, n2)

        if (n - n1 <= 2*window) or (n + 2*window >= n2):
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
        x, y = self.x[n1:n2], self.y[n1:n2]
        v = join_var(x, y)
        v_r = join_var(x[::-1], y[::-1])

        n_relative = np.argmin((v + v_r[::-1])[self.min_seg: -self.min_seg]) + self.min_seg
        return n1 + n_relative, v[n_relative], v_r[-n_relative]

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
