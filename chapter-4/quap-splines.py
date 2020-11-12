import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm


B = np.array(pd.read_table('spline3.tsv', delimiter=' '))

data = pd.read_table('cherry_blossoms_nomissing.tsv', delimiter=' ')

knots = np.array(pd.read_table('knots', delimiter=' ')).reshape(-1,)

year = np.array(data['year'])
doy = np.array(data['doy'])

with pm.Model() as normal_approximation:
	α = pm.Normal('α', mu=100, sigma=10)

	w = pm.Normal('w', mu=0, sigma=10, shape=B.shape[1])

	σ = pm.Exponential('σ', 1)

	μ = pm.Normal('height', mu=α + pm.math.dot(B, w), sigma=σ, observed=doy)

	mean_q = pm.find_MAP()

	hessian = pm.find_hessian(mean_q, vars=[α, w, σ])
	covariance_q = np.linalg.inv(hessian)
	std_q = np.sqrt(np.diag(covariance_q))

fig = plt.figure()
ax = fig.add_subplot(3,1,1)

for b in B.T:
    ax.plot(year, b, color='k')

ax = fig.add_subplot(3,1,2)
for ww, b in zip(mean_q['w'], B.T):
    ax.plot(year, ww*b, color='k')

ax = fig.add_subplot(3,1,3)
ax.scatter(year, doy, color='C0')

MAPs = np.hstack((mean_q['α'], mean_q['w'], mean_q['σ']))

samples = np.random.multivariate_normal(MAPs, covariance_q, size=10000)

mus = samples[:, 0] + B.dot(samples[:, 1:-1].T)

ci97 = np.percentile(mus, [1.5, 98.5], axis=1).T

ax.plot(year, ci97[:, 0], color='k', linestyle='--', alpha=0.7)
ax.plot(year, ci97[:, 1], color='k', linestyle='--', alpha=0.7)
ax.fill_between(year, ci97[:,0], ci97[:,1], color='k', alpha=0.2)
ax.set_ylim(80,130)