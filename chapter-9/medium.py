import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
from bayesian_routines import *

data = pd.read_csv('../chapter-8/rugged.csv', delimiter=';')
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

	trace = pm.sample(500, tune=500, chains=1, cores=1)

with pm.Model() as normalReg:

	a = pm.Normal('a', mu=1., sigma=0.1, shape=2)
	b = pm.Normal('b', mu=0., sigma=0.3, shape=2)
	sigma = pm.Uniform('sigma', 0., 1.)

	mu = pm.math.dot(africa_1h, a) + pm.math.dot(africa_1h, b) * (rugged_std - rMean)

	outcome = pm.Normal('logGDP', mu=mu, sigma=sigma, observed=log_gdp_std)

	trace1 = pm.sample(500, tune=500, chains=1, cores=1)

with pm.Model() as normalReg:

	a = pm.Normal('a', mu=1., sigma=0.1, shape=2)
	b = pm.Exponential('b', lam=0.3, shape=2)
	sigma = pm.Exponential('sigma', lam=1.)

	mu = pm.math.dot(africa_1h, a) + pm.math.dot(africa_1h, b) * (rugged_std - rMean)

	outcome = pm.Normal('logGDP', mu=mu, sigma=sigma, observed=log_gdp_std)

	trace2 = pm.sample(500, tune=500, chains=1, cores=1)


pm.traceplot(trace, var_names=['sigma']) 
pm.traceplot(trace1, var_names=['sigma']) 
# no real difference, exponential distribution allows for larger values than uniform
# but likelihood only "likes" small values


pm.traceplot(trace, var_names=['b']) 
pm.traceplot(trace2, var_names=['b']) 
# exponential prior doesn't allows negative values
# so the 'african' slope is all the way towards to zero

with pm.Model() as normalReg:

	a = pm.Normal('a', mu=1., sigma=0.1, shape=2)
	b = pm.Normal('b', mu=0., sigma=0.3, shape=2)
	sigma = pm.Exponential('sigma', lam=1.)

	mu = pm.math.dot(africa_1h, a) + pm.math.dot(africa_1h, b) * (rugged_std - rMean)

	outcome = pm.Normal('logGDP', mu=mu, sigma=sigma, observed=log_gdp_std)

	trace = pm.sample(500, tune=300, chains=1, cores=1)

