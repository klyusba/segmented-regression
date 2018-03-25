import numpy as np
import time
from segmentreg import SegmentedRegression
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = np.linspace(-10., 10., 100000)
    y = x/5 + np.tanh(x/2.5)/2

    plt.plot(x, y, '.', alpha=.5, color='#999999')

    for s in [1, 0.1, 0.05]:
        y_noisy = y + np.random.normal(scale=s, size=y.shape)
        m = SegmentedRegression()
        m.fit(x, y_noisy)
        y_ = m.predict(x)
        plt.plot(x, y_, label='s={}'.format(s))

    plt.legend()
    plt.show()
