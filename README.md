# segmented-regression
Discontinuous segmented linear regression

Model finds split points by greedy minimization of total variance. Linear trend is eliminated while variance is calculated.
Model require `numpy` and `bottleneck` for fast computations.

# Examples
```python
x = np.linspace(0., 5e8, 100000)
y = 1. - 1e-8*x
y[30000:] += 1.5
y[60000:] += 1.5
y[90000:] += 1.5
y += np.random.normal(scale=0.1, size=y.shape)

m = SegmentedRegression()
m.fit(x, y)
y_ = m.predict(x)
plt.plot(x, y, '.', alpha=.5, color='#999999')
plt.plot(x, y_, color='k')
```
![alt text](/img/simple_saw.png "Simple saw example")

```python
x = np.linspace(0., np.pi, 100000)
y = np.sin(x)

m = SegmentedRegression(eps=1e-4)
m.fit(x, y)
y_ = m.predict(x)
plt.plot(x, y, '.', alpha=.5, color='#999999')
plt.plot(x, y_, color='k')
```
![alt text](/img/sin.png "Sin example")
