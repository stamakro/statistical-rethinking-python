import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
'''
# 8H1 & 8H2
data = pd.read_csv('tulips.csv', sep=';')

water = np.array(data['water'])
water = water - np.mean(water)

shade = np.array(data['shade'])
shade = shade - np.mean(shade)

blooms = np.array(data['blooms'])
blooms = blooms / np.max(blooms)

bed2col = {'a':0, 'b':1, 'c':2}
bed1h = np.zeros((data.shape[0], 3), int)
for i, b in enumerate(np.array(data['bed'])):
	bed1h[i, bed2col[b]] = 1

np.random.seed(2021)

with pm.Model() as nobed:
	a = pm.Normal('a', mu=0.5, sigma=0.25)

	bw = pm.Normal('b_w', mu=0.0, sigma=0.25)
	bs = pm.Normal('b_s', mu=0.0, sigma=0.25)

	# interaction term
	bws = pm.Normal('b_ws', sigma=0.25)

	sigma = pm.Exponential('sigma', lam=1.)

	mu = a + bw * water + bs * shade + bws * water * shade

	outcome = pm.Normal('blooms', mu=mu, sigma=sigma, observed=blooms)
	prior_checks_nobed = pm.sample_prior_predictive(samples=50)
	trace_nobed = pm.sample(2000, tune=1000)


with pm.Model() as bed:
	#a = pm.Normal('a', mu=0.5, sigma=0.25)

	bw = pm.Normal('b_w', mu=0.0, sigma=0.25)
	bs = pm.Normal('b_s', mu=0.0, sigma=0.25)

	bedIntercepts = pm.Normal('a_bed', mu=0.5, sigma=0.25, shape=3)

	# interaction term
	bws = pm.Normal('b_ws', sigma=0.25)

	sigma = pm.Exponential('sigma', lam=1.)

	mu = pm.math.dot(bed1h, bedIntercepts) + bw * water + bs * shade + bws * water * shade

	outcome = pm.Normal('blooms', mu=mu, sigma=sigma, observed=blooms)
	prior_checks_bed = pm.sample_prior_predictive(samples=50)
	trace_bed = pm.sample(2000, tune=1000)



r = az.compare({'bed': trace_bed, 'nobed':trace_nobed}, 'WAIC')
print(r)

# first bed has 'significantly' lower intercept
az.plot_trace(trace_bed, var_names='a_bed')

fig = plt.figure()
ax = fig.add_subplot(111)
xs = [-1, 0, 1]

for i in range(3):
	ii = np.where(bed1h[:,i])[0]
	print(np.mean(blooms[ii]))
	ax.scatter(xs[i] * np.ones(ii.shape[0]), blooms[ii])

# there's indeed a big difference in means, bed a has much smaller mean, so knowing the bed can help the prediction
# probably a 3rd factor (i.e. neither water nor light) is different for bed a, making plant growth harder
'''

# 8H3
data = pd.read_csv('rugged.csv', delimiter=';')
data = data[data['rgdppc_2000'].notna()]

log_gdp = np.log(np.array(data['rgdppc_2000']))
rugged = np.array(data['rugged'])

log_gdp_std = log_gdp / np.mean(log_gdp)
rugged_std = rugged / np.max(rugged)

isInAfrica = np.array(data['cont_africa'])
africa_1h = np.zeros((isInAfrica.shape[0], 2))
for i, b in enumerate(isInAfrica):
	africa_1h[i, b] = 1

countries = np.array(data['country'])


rMean = np.mean(rugged_std)

with pm.Model() as normalReg:

	a = pm.Normal('a', mu=1., sigma=0.1, shape=2)
	b = pm.Normal('b', mu=0., sigma=0.3, shape=2)
	sigma = pm.Exponential('sigma', lam=1.)

	mu = pm.math.dot(africa_1h, a) + pm.math.dot(africa_1h, b) * (rugged_std - rMean)

	outcome = pm.Normal('logGDP', mu=mu, sigma=sigma, observed=log_gdp_std)

	trace = pm.sample(2000, tune=1000)

psis = az.loo(trace, pointwise=True)

sampleWeights = np.array(psis[7])
ind = np.argsort(sampleWeights)[::-1]


print(countries[ind[:5]])
# a) lesotho and seychelles have the biggest effect

with pm.Model() as robustReg:

	a = pm.Normal('a', mu=1., sigma=0.1, shape=2)
	b = pm.Normal('b', mu=0., sigma=0.3, shape=2)
	sigma = pm.Exponential('sigma', lam=1.)

	mu = pm.math.dot(africa_1h, a) + pm.math.dot(africa_1h, b) * (rugged_std - rMean)

	outcome = pm.StudentT('logGDP', nu=2., mu=mu, sigma=sigma, observed=log_gdp_std)

	traceR = pm.sample(2000, tune=1000)

psisR = az.loo(traceR, pointwise=True)

sampleWeightsR = np.array(psisR[7])
ind = np.argsort(sampleWeightsR)[::-1]

xx = np.linspace(-0.3, 0.6, 20)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(sampleWeights, sampleWeightsR)
ax.set_xlabel('Normal', fontsize=14)
ax.set_ylabel('Robust', fontsize=14)
ax.set_title('PSIS sample weights')

ax.plot(xx, xx, 'k--')

# the influence of outliers is greatly diminished
