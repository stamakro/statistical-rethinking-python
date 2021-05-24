import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pymc3 as pm

# 12H1

data = pd.read_csv('Hurricanes.csv', delimiter=';')
deaths = np.array(data['deaths'])
fem = np.array(data['femininity'])

ss = StandardScaler()
fem = ss.fit_transform(fem.reshape(-1,1)).reshape(-1,)

with pm.Model() as m0:
	a = pm.Normal('a', 0, 1.5)

	l = pm.math.exp(a)

	d = pm.Poisson('D', mu=l, observed=deaths)

	trace0 = pm.sample(500, chains=4, tune=1000)


with pm.Model() as m1:
	a = pm.Normal('a', 0, 1.5)
	bF = pm.Normal('b', 0, 0.5)

	mm = a + bF * fem

	l = pm.math.exp(mm)

	d = pm.Poisson('D', mu=l, observed=deaths)

	trace1 = pm.sample(500, chains=4, tune=1000)


tracePost = pm.sample_posterior_predictive(trace1, model=m1)
posteriorDeaths = tracePost['D']

alphas = trace1['a']
betas = trace1['b']


lamdas = (alphas +  np.tile(fem, (2000, 1)).T * betas).T
lamdas = np.exp(lamdas)
mu = np.mean(lamdas, axis=0)

muCI = np.percentile(lamdas, [4.5, 95.5], axis=0).T

ind = np.argsort(fem)

muCI = muCI[ind]

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(fem, deaths, color='C0', edgecolor='k')
ax.plot(fem[ind], mu[ind], color='k')
ax.fill_between(fem[ind], muCI[:,1], muCI[:,0], color='k', alpha=0.5)

ppCI = np.percentile(posteriorDeaths, [4.5, 95.5], axis=0).T
ppCI = ppCI[ind]


ax.fill_between(fem[ind], ppCI[:,1], ppCI[:,0], color='k', alpha=0.1)

# coefficient of femininity is reliably above 0, but model is really bad at predicting very large death counts

# 12H2

with pm.Model() as m2:
	a = pm.Normal('a', 0, 1.5)
	bF = pm.Normal('b', 0, 0.5)
	phi = pm.Exponential('phi', 1.)

	mm = a + bF * fem

	l = pm.math.exp(mm)

	d = pm.NegativeBinomial('D', mu=l, alpha=phi, observed=deaths)

	trace2 = pm.sample(500, chains=4, tune=1000)


