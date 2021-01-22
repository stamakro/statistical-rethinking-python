import numpy as np
import pandas as pd
import pymc3 as pm
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from bayesian_routines import *

'''
data = pd.read_csv('../chapter-5/WaffleDivorce.csv', delimiter=';')


# 6Η1
# based on dag of page 187

W = standardize(data['WaffleHouses'])
M = standardize(data['Marriage'])
D = standardize(data['Divorce'])
A = standardize(data['MedianAgeMarriage'])
S = data['South']


with pm.Model() as model:
	a = pm.Normal('α', mu=0, sigma=1.)
	aS = pm.Normal('α_S', mu=0, sigma=1.)
	aN = pm.Normal('α_N', mu=0, sigma=1.)
	bW = pm.Normal('β_W', mu=0, sigma=1)

	s = pm.Exponential('σ', lam=1.)
	#bS = pm.Normal('β_S', mu=0, sigma=1)

	m = a + bW * W + aN * (1-S) + aS * S

	outcome = pm.Normal('D', mu=m, sigma=s, observed=D)

	mean_q = pm.find_MAP()

	hessian = pm.find_hessian(mean_q, vars=[a, aS, aN, bW, s])
	covariance_q = np.linalg.inv(hessian)
	std_q = np.sqrt(np.diag(covariance_q))

samples = np.random.multivariate_normal([mean_q['α'], mean_q['α_S'], mean_q['α_N'], mean_q['β_W'], mean_q['σ']], covariance_q, size=10000)

xx = np.linspace(-2.5,2.5,300)
fig = plt.figure()
ax = fig.add_subplot(111)

dW = gaussian_kde(samples[:,3])
dS = gaussian_kde(samples[:,1] - samples[:,2])

ax.plot(xx, dW(xx), color='C0', label='#WaffleHouses')
ax.plot(xx, dS(xx), color='C1', label='South')
plt.legend()

precis(samples, ['α', 'α_S', 'α_N', 'β_W', 'σ'])


# the posterior


# 6Η2

# implied cond. dep.
# D _|_ S | A, M, W
# A _|_ W | S		ok (posterior of W coef has decent mass on both sides of zero)
# M _|_ W | S		ok (posterior of W coef has decent mass on both sides of zero)

with pm.Model() as model:
	a = pm.Normal('α', mu=0, sigma=1.)
	aS = pm.Normal('α_S', mu=0, sigma=1.)
	aN = pm.Normal('α_N', mu=0, sigma=1.)
	bW = pm.Normal('β_W', mu=0, sigma=1)

	s = pm.Exponential('σ', lam=1.)
	#bS = pm.Normal('β_S', mu=0, sigma=1)

	m = a + bW * W + aN * (1-S) + aS * S

	outcome = pm.Normal('A', mu=m, sigma=s, observed=A)

	mean_q = pm.find_MAP()

	hessian = pm.find_hessian(mean_q, vars=[a, aS, aN, bW, s])
	covariance_q = np.linalg.inv(hessian)
	std_q = np.sqrt(np.diag(covariance_q))

samples = np.random.multivariate_normal([mean_q['α'], mean_q['α_S'], mean_q['α_N'], mean_q['β_W'], mean_q['σ']], covariance_q, size=10000)


xx = np.linspace(-2.5,2.5,300)
fig = plt.figure()
ax = fig.add_subplot(111)

dW = gaussian_kde(samples[:,3])
dS = gaussian_kde(samples[:,1] - samples[:,2])

ax.plot(xx, dW(xx), color='C0', label='#WaffleHouses')
ax.plot(xx, dS(xx), color='C1', label='South')
plt.legend()

precis(samples, ['α', 'α_S', 'α_N', 'β_W', 'σ'])



with pm.Model() as model:
	a = pm.Normal('α', mu=0., sigma=1.)
	aS = pm.Normal('α_S', mu=0., sigma=1.)
	aN = pm.Normal('α_N', mu=0., sigma=1.)
	bW = pm.Normal('β_W', mu=0., sigma=1.)
	bA = pm.Normal('β_A', mu=0., sigma=1.)
	bM = pm.Normal('β_M', mu=0., sigma=1.)


	s = pm.Exponential('σ', lam=1.)
	#bS = pm.Normal('β_S', mu=0, sigma=1)

	m = a + bW * W + bA * A + bM * M + aN * (1-S) + aS * S

	outcome = pm.Normal('D', mu=m, sigma=s, observed=D)

	mean_q = pm.find_MAP()

	hessian = pm.find_hessian(mean_q, vars=[a, aS, aN, bW, bA, bM, s])
	covariance_q = np.linalg.inv(hessian)
	std_q = np.sqrt(np.diag(covariance_q))

samples = np.random.multivariate_normal([mean_q['α'], mean_q['α_S'], mean_q['α_N'], mean_q['β_W'], mean_q['β_A'], mean_q['β_M'], mean_q['σ']], covariance_q, size=10000)

precis(samples, ['α', 'α_S', 'α_N', 'β_W', 'β_A', 'β_M', 'σ'])
'''

data = pd.read_csv('foxes.csv', delimiter=';')

area = standardize(np.array(data['area']))
avgfood = standardize(np.array(data['avgfood']))
weight = standardize(np.array(data['weight']))
groupsize = standardize(np.array(data['groupsize']))


# 6H3

# prior predictive simulation, mean and variance 0, 0.4 for both weight and intercept seems reasonable, lambda 0.5 also
mp = 0.
sp = 0.4
lp = 0.5

mmy = np.min(weight)
mmx = np.min(area)

MMy = np.max(weight)
MMx = np.max(area)

xx = np.linspace(mmx, MMx, 100)

N = 1000

priorSamples = np.random.multivariate_normal(np.zeros(2), sp * np.eye(2), N)
priorS = np.random.exponential(scale=0.5, size=N)

fig = plt.figure()
ax = fig.add_subplot(121)
for i in range(100):
	ax.plot(xx, priorSamples[i,0] + priorSamples[i,1] * xx, color='k', alpha=0.4)

ax.axhline(mmy, color='r')
ax.axhline(MMy, color='r')

ax = fig.add_subplot(122)
ypred = np.random.normal(priorSamples[:,0] + priorSamples[:,1] * np.tile(xx, (N,1)).T, priorS)
ll = np.percentile(ypred, 5.5, axis=1)
uu = np.percentile(ypred, 94.5, axis=1)

ax.scatter(area, weight)
ax.fill_between(xx, ll, uu, color='k', alpha=0.2)


# no need for other covariates
with pm.Model() as foxy:
	a = pm.Normal('a', mu=mp, sigma=sp)
	b = pm.Normal('b', mu=mp, sigma=sp)
	s = pm.Exponential('s', lam=lp)

	mu = a + b * area

	W = pm.Normal('w', mu=mu, sigma=s, observed=weight)


	mean_q = pm.find_MAP()

	hessian = pm.find_hessian(mean_q, vars=[a, b, s])
	covariance_q = np.linalg.inv(hessian)
	std_q = np.sqrt(np.diag(covariance_q))

samples = np.random.multivariate_normal([mean_q['a'], mean_q['b'], mean_q['s']], covariance_q, size=10000)
precis(samples, ['a', 'b', 's'])
print(np.mean(samples[:, 1] > 0))

# there isn't really a big effect;  p(b > 0) ~= 0.57

# 6H4

# no need for other covariates, arrow enters avgfood from area, but that path is blocked by conditioning on avgfood
with pm.Model() as foxy:
	a = pm.Normal('a', mu=mp, sigma=sp)
	b = pm.Normal('b', mu=mp, sigma=sp)
	s = pm.Exponential('s', lam=lp)

	mu = a + b * avgfood

	W = pm.Normal('w', mu=mu, sigma=s, observed=weight)


	mean_q = pm.find_MAP()

	hessian = pm.find_hessian(mean_q, vars=[a, b, s])
	covariance_q = np.linalg.inv(hessian)
	std_q = np.sqrt(np.diag(covariance_q))

samples = np.random.multivariate_normal([mean_q['a'], mean_q['b'], mean_q['s']], covariance_q, size=10000)
precis(samples, ['a', 'b', 's'])
print(np.mean(samples[:, 1] > 0))

# no effect;  p(b > 0) ~= 0.4


# 6H5

# we need to adjust for avgfood, incoming arrow and not collider
with pm.Model() as foxy:
	a = pm.Normal('a', mu=mp, sigma=sp)
	b = pm.Normal('b', mu=mp, sigma=sp)
	bcov = pm.Normal('b2', mu=mp, sigma=sp)
	s = pm.Exponential('s', lam=lp)

	mu = a + b * groupsize + bcov * avgfood

	W = pm.Normal('w', mu=mu, sigma=s, observed=weight)


	mean_q = pm.find_MAP()

	hessian = pm.find_hessian(mean_q, vars=[a, b, bcov, s])
	covariance_q = np.linalg.inv(hessian)
	std_q = np.sqrt(np.diag(covariance_q))

samples = np.random.multivariate_normal([mean_q['a'], mean_q['b'], mean_q['b2'], mean_q['s']], covariance_q, size=10000)
precis(samples, ['a', 'grpsz', 'avgf', 's'])

# now both groupsize and avgfood weights are non-zero with probability >0.99
# it seems that avgfood increases the weight, but also increases the groupsize, which leaves less food for each fox in the group.
# so the overall effect of avgfood is almost 0, but given a specific groupsize, more food clearly leads to more weight




plt.show()
