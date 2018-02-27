# segmented-regression
Discontinuous segmented linear regression

Model finds split points by greedy minimization of total variance. Linear trend is eliminated before variance is calculated.
Model is fast and require only `numpy`.

# Examples
```python
x = np.linspace(0., 500., 100000)
y = 100. - 1.*x
y[30000:] += 150
y[60000:] += 150
y[90000:] += 150
y += np.random.normal(scale=10, size=y.shape)

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

m = SegmentedRegression(eps=1e-3)
m.fit(x, y)
y_ = m.predict(x)
plt.plot(x, y, '.', alpha=.5, color='#999999')
plt.plot(x, y_, color='k')
```
![alt text](/img/sin.png "Sin example")
