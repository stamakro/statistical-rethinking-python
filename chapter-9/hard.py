import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
from bayesian_routines import *
import arviz as az
import pandas as pd

# 9H1
'''with pm.Model() as test:
	a = pm.Normal('a', 0, 1)
	b = pm.Cauchy('b', 0, 1)
'''

'''
# 9H2

data = pd.read_csv('../chapter-5/WaffleDivorce.csv', delimiter=';')

divorce = standardize(np.array(data.Divorce))
marriage = standardize(np.array(data.Marriage))
ageAtMarriage = standardize(np.array(data.MedianAgeMarriage))


with pm.Model() as model1:
	a = pm.Normal('a', 0, 0.2)
	bA = pm.Normal('b', 0, 0.5)
	s = pm.Exponential('s', lam=1.)

	m = a + bA * ageAtMarriage

	d = pm.Normal('div', m, s, observed=divorce)

	trace1 = pm.sample(1000, tune=500, chains=1, cores=1)


with pm.Model() as model2:
	a = pm.Normal('a', 0, 0.2)
	bM = pm.Normal('b', 0, 0.5)
	s = pm.Exponential('s', 1.)

	m = a + bM * marriage

	d = pm.Normal('div', m, s, observed=divorce)

	trace2 = pm.sample(1000, tune=500, chains=1, cores=1)


with pm.Model() as model3:
	a = pm.Normal('a', 0, 0.2)
	bM = pm.Normal('bM', 0, 0.5)
	bA = pm.Normal('bA', 0, 0.5)
	s = pm.Exponential('s', 1.)

	m = a + bM * marriage + bA * ageAtMarriage

	d = pm.Normal('div', m, s, observed=divorce)

	trace3 = pm.sample(1000, tune=500, chains=1, cores=1)

comp = az.compare({'a': trace1, 'm': trace2, 'a+m': trace3})
print(comp)

# marriage rate is the worst, age is the best, because it has fewer features?
# difference in waic between a and a+m is 1.96 +/- 0.91, not so big
'''

# 9H3
np.random.seed(92351)

N = 100
height = np.random.normal(10, 2, size=N)
leg_prop = 0.4 + np.random.rand(N) * 10.
leg_left = leg_prop * height + np.random.normal(0, 0.02, N)
leg_right = leg_prop * height + np.random.normal(0, 0.02, N)


with pm.Model() as m1:
	a = pm.Normal('a', 10, 100)
	bl = pm.Normal('bl', 2, 10)
	br = pm.Normal('br', 2, 10)

	s = pm.Exponential('s', lam=1)

	mu  = a + bl * leg_left + br * leg_right

	h = pm.Normal('h', mu, s, observed=height)

	trace1 = pm.sample(1000, tune=500, chains=1, cores=1, start={'a': 10, 'bl':0, 'br':0.1, 's': 1})


with pm.Model() as m2:
	a = pm.Normal('a', 10, 100)
	bl = pm.Normal('bl', 2, 10)
	br = pm.Bound(pm.Normal, lower=0.)('br', mu=2., sigma=10.)

	s = pm.Exponential('s', lam=1)

	mu  = a + bl * leg_left + br * leg_right

	h = pm.Normal('h', mu, s, observed=height)

	trace2 = pm.sample(1000, tune=500, chains=1, cores=1, start={'a': 10, 'bl':0, 'br':0.1, 's': 1})

# sampling is horrendously slow; because of the colinearity?
# 
pm.traceplot(trace1, var_names=['bl', 'br'])
pm.traceplot(trace2, var_names=['bl', 'br'])

fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.scatter(trace1['bl'], trace1['br'])
ax.set_xlabel('bl')
ax.set_ylabel('br')


ax = fig.add_subplot(1,2,2)
ax.scatter(trace2['bl'], trace2['br'])
ax.set_xlabel('bl')
ax.set_ylabel('br')

# bl is now always negative, while before it was always positive

# 9H4
res = az.compare({'U': trace1, 'C': trace2})
print(res)

# no real difference in waic


# 9H5
# ?


'''
# 9H6
from scipy.stats import binom, gaussian_kde, beta

N = 3000

ps = np.zeros(N)

proposalSigma = 0.1

currentP = np.random.rand()
ps[0] = currentP

currentPosterior = binom.pmf(6, 9, ps[0])
naccepted = 0

for i in range(1, N):
	pnew = ps[i-1] + np.random.normal(0, proposalSigma)

	if pnew < 0 or pnew > 1:
		prior = 0.
	else:
		prior = 1.

	postNew = prior * binom.pmf(6, 9, pnew) 

	ratio = postNew / currentPosterior
	
	if np.random.rand() < ratio:
		currentPosterior = postNew
		currentP = pnew
		naccepted += 1

	ps[i] = currentP


print('acceptance rate: %f' % (100 * naccepted / (N-1)))

xx = np.linspace(0,1,200)
fig = plt.figure()
ax = fig.add_subplot(111)
d = gaussian_kde(ps)
ax.plot(xx, d(xx), color='k', label='mcmc')
ax.plot(xx, beta.pdf(xx, 7, 4), color='C1', label='conjugate')

plt.legend()
'''


plt.show()

# 9H7
# too hard
