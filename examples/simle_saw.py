import numpy as np
import time
from segmentreg import SegmentedRegression
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = np.linspace(0., 500., 100000)
    y = 100. - 1.*x
    y[30000:] += 150
    y[60000:] += 150
    y[90000:] += 150
    y += np.random.normal(scale=10, size=y.shape)

    t = time.time()
    m = SegmentedRegression()
    m.fit(x, y)
    t = time.time() - t
    print('Splits: [{}], time: {:.1f}ms'.format(', '.join(str(a) for a in m.splits_), t * 1000))

    plt.plot(x, y, '.', alpha=.5, color='#999999')
    for x1, x2, (k, b) in zip([0, ]+m.splits_, m.splits_+[len(y)], m.lines):
        plt.plot(x[x1:x2], k * x[x1:x2] + b, color='k')
    plt.show()
