import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm

data = pd.read_csv('Howell1.csv', delimiter=';')

#d2 = data[data['age'] >= 18]
d2 = data

height = d2['height']
weight = d2['weight']

#optionally downsample data
#ind = np.random.permutation(height.shape[0])
#height = np.array(height)[ind]
#weight = np.array(weight)[ind]

weightBar = np.mean(weight)
'''
#----------------------------------#
#	linear fit		   #
#----------------------------------#
with pm.Model() as normal_approximation:
	alpha = pm.Normal('alpha', mu=178, sigma=20)	
	beta = pm.Lognormal('beta', mu=0, sigma=1)

	s = pm.Uniform('sigma', 0, 50)

	h = pm.Normal('height', mu=alpha + beta*(weight-weightBar), sigma=s, observed=height)

	mean_q = pm.find_MAP()

	hessian = pm.find_hessian(mean_q, vars=[alpha, beta, s])


	covariance_q = np.linalg.inv(hessian)

	std_q = np.sqrt(np.diag(covariance_q))


#draw samples from posterior
samples = np.random.multivariate_normal([mean_q['alpha'], mean_q['beta'], mean_q['sigma']], covariance_q, size=10000)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(weight, height, color='w', edgecolor='C0', alpha=0.5)

a_map, b_map, s_map = np.mean(samples,0)

ww = np.arange(0, 65)
ax.plot(ww, a_map + b_map * (ww-weightBar), color='k')

mus = samples[:, 0 ] + samples[:, 1] * np.tile(ww.reshape(1,ww.shape[0]) - weightBar, (samples.shape[0],1)).T

ci89 = np.percentile(mus, [5.5, 94.5], axis=1).T


ax.plot(ww, ci89[:,0], color='k', linestyle='--', alpha=0.7)
ax.plot(ww, ci89[:,1], color='k', linestyle='--', alpha=0.7)

ax.fill_between(ww, ci89[:,0], ci89[:,1], color='k', alpha=0.2)

posteriorDraws = np.random.normal(mus, samples[:,2])
posteriorCi89 = np.percentile(posteriorDraws, [5.5, 94.5], axis=1).T


ax.fill_between(ww, posteriorCi89[:,0], posteriorCi89[:,1], color='k', alpha=0.1)
'''

#----------------------------------#
#	polynomial fit		   #
#----------------------------------#
from sklearn.preprocessing import StandardScaler
weight = np.array(weight)
height = np.array(height)

mm = np.mean(weight)
ss = np.std(weight, ddof=1)

weightS = (weight - mm) / ss
#preproc = StandardScaler()
#weightS = preproc.fit_transform(weight.reshape(-1,1))
#this uses biased variance estimation

weightS2 = weightS ** 2


with pm.Model() as normal_approximation:
	alpha = pm.Normal('alpha', mu=178, sigma=20)	
	beta1 = pm.Lognormal('beta1', mu=0, sigma=1)
	beta2 = pm.Normal('beta2', mu=0, sigma=1)

	s = pm.Uniform('sigma', 0, 50)

	h = pm.Normal('height', mu=alpha + beta1* weightS + beta2 * weightS2, sigma=s, observed=height)

	mean_q = pm.find_MAP()

	hessian = pm.find_hessian(mean_q, vars=[alpha, beta1, beta2, s])


	covariance_q = np.linalg.inv(hessian)

	std_q = np.sqrt(np.diag(covariance_q))


plt.show()
