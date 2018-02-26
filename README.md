# segmented-regression
Discontinuous segmented linear regression

Model finds split points by greedy minimization of variance reduction. Linear trend is eliminated before variance is calculated.
Model if fast and require only `numpy`.

# Example
```python
x = np.linspace(0., 500., 100000)
y = 100. - 1.*x
y[30000:] += 150
y[60000:] += 150
y[90000:] += 150
y += np.random.normal(scale=10, size=y.shape)

m = SegmentedRegression()
m.fit(x, y)
```
![alt text](/img/simple_saw.png "Simple saw example")
