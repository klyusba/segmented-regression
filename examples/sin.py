import numpy as np
import time
from segmentreg import SegmentedRegression
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = np.linspace(0., np.pi, 100000)
    y = np.sin(x)

    t = time.time()
    m = SegmentedRegression(eps=1e-3)
    m.fit(x, y)
    t = time.time() - t
    print('Segments: {}, time: {:.1f}ms'.format(len(m.segments_), t * 1000))

    plt.plot(x, y, '.', alpha=.5, color='#999999')
    y_ = m.predict(x)
    plt.plot(x, y_, color='k')
    plt.show()
