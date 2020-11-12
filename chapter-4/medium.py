import numpy as np
import pymc3 as pm
import pandas as pd
import matplotlib.pyplot as plt


#4.1
N = 10000
m = np.random.normal(0, 10, N)
with pm.Model() as fml:
    s = pm.distributions.continuous.Exponential('s', 1).random(size=N)

y = np.random.normal(m, s)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(y, bins=50, edgecolor='k')


#4.2
with pm.Model() as ex4dot2:
    m = pm.Normal('μ', mu=0, sigma=10)
    s = pm.Exponential('σ', 1)

    y = pm.Normal('y', mu=m, sigma=s, observed=True)

#4.3
"""
y ~ Normal(μ, σ)
μ = α + β * x
α ~ Normal(0, 10)
β ~ Uniform(0, 1)
σ ~ Exponential(1)
"""

#4.4
"""
t: time in years
h: height in cm

h ~ Normal(μ, σ)
μ = α + β * t

α ~ Normal(50, 2)  
(at birth mean height is 50cm, with values 47-53 considered normal, being less strict have 2σ in range 46-54)

β ~ Normal(6.7, 3) 
(assuming mean adult height 170, a person grows 120cm in 18 years, mean is 6.667cm/year, 
σ so that β has small chance of being negative)

σ ~ Exponential(0.2) 
(average deviation from mean 5cm)
"""


#4.5-4.6
"""
4.5 perhaps a log normal for β

4.6 σ uniform betwen [0, 8]?

"""


#4.7

data = pd.read_csv('Howell1.csv', delimiter=';')

d2 = data[data['age'] >= 18]

height = d2['height']
weight = d2['weight']

weightBar = np.mean(weight)

with pm.Model() as centered:
	alpha = pm.Normal('alpha', mu=178, sigma=20)
	beta = pm.Lognormal('beta', mu=0, sigma=1)

	s = pm.Uniform('sigma', 0, 50)

	h = pm.Normal('height', mu=alpha + beta*(weight-weightBar), sigma=s, observed=height)

	mean_q = pm.find_MAP()

	hessian = pm.find_hessian(mean_q, vars=[alpha, beta, s])


	covariance_q = np.linalg.inv(hessian)

	std_q = np.sqrt(np.diag(covariance_q))



with pm.Model() as uncentered:
	alpha2 = pm.Normal('alpha', mu=178, sigma=20)
	beta2 = pm.Lognormal('beta', mu=0, sigma=1)

	s2 = pm.Uniform('sigma', 0, 50)

	h2 = pm.Normal('height', mu=alpha2 + beta2*weight, sigma=s2, observed=height)

	mean_q2 = pm.find_MAP()

	hessian2 = pm.find_hessian(mean_q2, vars=[alpha2, beta2, s2])

	covariance_q2 = np.linalg.inv(hessian2)

	std_q2 = np.sqrt(np.diag(covariance_q2))

corr = (covariance_q / std_q).T / std_q
corr2 = (covariance_q2 / std_q2).T / std_q2
#correlation helps to see even more clearly, α, β are highly correlated in the 2nd case, while almost uncorrelated in the first



#posterior predictive distribution of 1st model
samples = np.random.multivariate_normal([mean_q['alpha'], mean_q['beta'], mean_q['sigma']], covariance_q, size=10000)

fig = plt.figure()
ax = fig.add_subplot(121)
ax.scatter(weight, height, color='w', edgecolor='C0', alpha=0.5)

a_map, b_map, s_map = np.mean(samples,0)

ww = np.arange(25, 65)
ax.plot(ww, a_map + b_map * (ww-weightBar), color='k')

mus = samples[:, 0 ] + samples[:, 1] * np.tile(ww.reshape(1,ww.shape[0]) - weightBar, (samples.shape[0],1)).T

ci89 = np.percentile(mus, [5.5, 94.5], axis=1).T


ax.plot(ww, ci89[:,0], color='k', linestyle='--', alpha=0.7)
ax.plot(ww, ci89[:,1], color='k', linestyle='--', alpha=0.7)

ax.fill_between(ww, ci89[:,0], ci89[:,1], color='k', alpha=0.2)

posteriorDraws = np.random.normal(mus, samples[:,2])
posteriorCi89 = np.percentile(posteriorDraws, [5.5, 94.5], axis=1).T


ax.fill_between(ww, posteriorCi89[:,0], posteriorCi89[:,1], color='k', alpha=0.1)


#posterior predictive distribution of 2nd model
samples = np.random.multivariate_normal([mean_q2['alpha'], mean_q2['beta'], mean_q2['sigma']], covariance_q2, size=10000)

ax = fig.add_subplot(122)
ax.scatter(weight, height, color='w', edgecolor='C0', alpha=0.5)

a_map, b_map, s_map = np.mean(samples,0)

ww = np.arange(25, 65)
ax.plot(ww, a_map + b_map * ww, color='k')

mus = samples[:, 0 ] + samples[:, 1] * np.tile(ww.reshape(1,ww.shape[0]), (samples.shape[0],1)).T

ci89 = np.percentile(mus, [5.5, 94.5], axis=1).T


ax.plot(ww, ci89[:,0], color='k', linestyle='--', alpha=0.7)
ax.plot(ww, ci89[:,1], color='k', linestyle='--', alpha=0.7)

ax.fill_between(ww, ci89[:,0], ci89[:,1], color='k', alpha=0.2)

posteriorDraws = np.random.normal(mus, samples[:,2])
posteriorCi89_2 = np.percentile(posteriorDraws, [5.5, 94.5], axis=1).T


ax.fill_between(ww, posteriorCi89_2[:,0], posteriorCi89_2[:,1], color='k', alpha=0.1)
#I don't see big differences in posterior predictive(?)
#It seems that the 2nd model predicts lower values than the 1st to the left of the mean,
#and higher to the right of the mean. But why?





plt.show()