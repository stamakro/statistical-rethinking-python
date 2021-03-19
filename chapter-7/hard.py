import numpy as np
import pymc3 as pm
import pandas as pd
import arviz as az
from scipy import stats
from scipy.special import logsumexp
from bayesian_routines import standardize
import matplotlib.pyplot as plt

np.random.seed(20897234)
# 7H1
data = pd.read_csv('Laffer.csv', delimiter=';')

rate = standardize(np.array(data['tax_rate']))
revenue = standardize(np.array(data['tax_revenue']))


n_samples = 2000 // 4
n_tuning = 1000

with pm.Model() as linear:
	#a = pm.Normal('b0', mu=0., sigma=0.5)
	b = pm.Normal('b1', mu=0., sigma=0.5)

	s = pm.Exponential('s', lam=0.5)

	m = b * rate

	r = pm.Normal('r', mu=m, sigma=s, observed=revenue)

	traceLinear = pm.sample(n_samples, tune=n_tuning)


with pm.Model() as quadratic:
	a = pm.Normal('b0', mu=0., sigma=0.5)
	b = pm.Normal('b1', mu=0., sigma=0.5)
	c = pm.Normal('b2', mu=0., sigma=0.5)

	s = pm.Exponential('s', lam=0.5)

	m = a + b * rate + c * (rate ** 2)

	r = pm.Normal('r', mu=m, sigma=s, observed=revenue)

	traceQuadratic = pm.sample(n_samples, tune=n_tuning)


with pm.Model() as cubic:
	a = pm.Normal('b0', mu=0., sigma=0.5)
	b = pm.Normal('b1', mu=0., sigma=0.5)
	c = pm.Normal('b2', mu=0., sigma=0.5)
	d = pm.Normal('b3', mu=0., sigma=0.5)

	s = pm.Exponential('s', lam=0.5)

	m = a + b * rate + c * (rate ** 2) + d * (rate ** 3)

	r = pm.Normal('r', mu=m, sigma=s, observed=revenue)

	traceCubic = pm.sample(n_samples, tune=n_tuning)


r = az.compare({'L': traceLinear, 'Q':traceQuadratic, 'C':traceCubic}, 'WAIC')
print(r)

'''
#------------------------------------------------------------------------------------------------------------------
# 7H2
ww = az.waic(traceLinear, pointwise=True)

sInd = np.argmax(revenue)
sampleImportance = np.array(ww[6])


with pm.Model() as linearRobust:
	a = pm.Normal('b0', mu=0., sigma=0.5)
	b = pm.Normal('b1', mu=0., sigma=0.5)

	s = pm.Exponential('s', lam=0.5)

	m = a + b * rate

	r = pm.StudentT('r', nu=1., mu=m, sigma=s, observed=revenue)

	traceLinearRobust = pm.sample(n_samples, tune=n_tuning)

ww = az.waic(traceLinearRobust, pointwise=True)

sampleImportance2 = np.array(ww[6])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(sampleImportance, sampleImportance2)
ax.set_xlabel('Normal')
ax.set_ylabel('Student t')
ax.set_xlim(0, 25)
ax.set_ylim(0, 25)

# effect is reduce ~by half, but still it is pretty big, much bigger than the other points

#------------------------------------------------------------------------------------------------------------------
# 7H3
def entropy(x):
	return - np.sum(x * np.log2(x))

def kldivergence(p,q):
	return np.sum(p * np.log2(p / q))

d = np.array([[0.2, 0.2, 0.2, 0.2, 0.2], [0.8, 0.1, 0.05, 0.025, 0.025], [0.05, 0.15, 0.7, 0.05, 0.05]])

e = np.zeros(d.shape[0])
for i, island in enumerate(d):
	e[i] = entropy(island)

kls = np.zeros((3,3))

for i in range(3):
	for j in range(3):
		if i != j:
			kls[i,j] = kldivergence(d[i], d[j])


# entropy decreases from 1, 3, 2. 1 is the most 'random'. It is very much expecting species A very often
# more entropy means less KL divergence to other distributions, because it's harder to be surprised by what you see

plt.show()

#------------------------------------------------------------------------------------------------------------------
# 7H5

data = pd.read_csv('../chapter-6/foxes.csv', delimiter=';')

area = standardize(np.array(data['area']))
avgfood = standardize(np.array(data['avgfood']))
weight = standardize(np.array(data['weight']))
groupsize = standardize(np.array(data['groupsize']))


mp = 0.
sp = 0.4
lp = 0.5

n_samples = 2000 // 4
n_tuning = 1000

with pm.Model() as fox1:
	w_f = pm.Normal('wf', mu=mp, sigma=sp)
	w_g = pm.Normal('wg', mu=mp, sigma=sp)
	w_a = pm.Normal('wa', mu=mp, sigma=sp)
	w_0 = pm.Normal('w0', mu=mp, sigma=sp)

	s = pm.Exponential('s', lam=lp)

	mu = w_0 + w_f * avgfood + w_g * groupsize + w_a * area

	W = pm.Normal('w', mu=mu, sigma=s, observed=weight)

	trace1 = pm.sample(n_samples, tune=n_tuning)


with pm.Model() as fox2:
	w_f = pm.Normal('wf', mu=mp, sigma=sp)
	w_g = pm.Normal('wg', mu=mp, sigma=sp)
	w_0 = pm.Normal('w0', mu=mp, sigma=sp)

	s = pm.Exponential('s', lam=lp)

	mu = w_0 + w_f * avgfood + w_g * groupsize 
	W = pm.Normal('w', mu=mu, sigma=s, observed=weight)

	trace2 = pm.sample(n_samples, tune=n_tuning)


with pm.Model() as fox3:
	w_g = pm.Normal('wg', mu=mp, sigma=sp)
	w_a = pm.Normal('wa', mu=mp, sigma=sp)
	w_0 = pm.Normal('w0', mu=mp, sigma=sp)

	s = pm.Exponential('s', lam=lp)

	mu = w_0 + w_g * groupsize + w_a * area

	W = pm.Normal('w', mu=mu, sigma=s, observed=weight)

	trace3 = pm.sample(n_samples, tune=n_tuning)


with pm.Model() as fox4:
	w_f = pm.Normal('wf', mu=mp, sigma=sp)
	w_0 = pm.Normal('w0', mu=mp, sigma=sp)

	s = pm.Exponential('s', lam=lp)

	mu = w_0 + w_f * avgfood 
	W = pm.Normal('w', mu=mu, sigma=s, observed=weight)

	trace4 = pm.sample(n_samples, tune=n_tuning)


with pm.Model() as fox5:
	w_a = pm.Normal('wa', mu=mp, sigma=sp)
	w_0 = pm.Normal('w0', mu=mp, sigma=sp)

	s = pm.Exponential('s', lam=lp)

	mu = w_0 + w_a * area

	W = pm.Normal('w', mu=mu, sigma=s, observed=weight)

	trace5 = pm.sample(n_samples, tune=n_tuning)


r = az.compare({'m1': trace1, 'm2':trace2, 'm3':trace3, 'm4': trace4, 'm5': trace5}, 'WAIC')
'''
# model with all variables is the best
# ignoring group size is really bad, because it has a negative effect from the other two variables
# using area instead of avgfood is slightly worse and the difference is more consistent (smaller std error)
# probably because area is an indirect cause of weight, while food is direct?


