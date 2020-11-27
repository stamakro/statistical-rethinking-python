import numpy as np
import pymc3 as pm
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Howell1.csv', delimiter=';')
#4.1
'''


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


samples = np.random.multivariate_normal([mean_q['alpha'], mean_q['beta'], mean_q['sigma']], covariance_q, size=10000)

a_map, b_map, s_map = np.mean(samples,0)

ww = np.array([46.95, 43.72, 64.78, 32.59, 54.63])
pred = a_map + b_map * (ww-weightBar)

mus = samples[:, 0 ] + samples[:, 1] * np.tile(ww.reshape(1,ww.shape[0]) - weightBar, (samples.shape[0],1)).T

ci89 = np.percentile(mus, [5.5, 94.5], axis=1).T

for w, hp, cc in zip(ww, pred, ci89):
    print('%.2f\t%.2f\t[%.2f, %.2f]' % (w, hp, cc[0], cc[1]))


#4.2
d2 = data[data['age'] < 18]

height = d2['height']
weight = d2['weight']

weightBar = np.mean(weight)

with pm.Model() as normal_approximation:
	alpha = pm.Normal('alpha', mu=120, sigma=20)
	beta = pm.Lognormal('beta', mu=0, sigma=1)

	s = pm.Uniform('sigma', 0, 20)

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
ax.set_title('4.2')
print('%f kg / 10 cm' % 10 * b_map)

# not a good fit,
# changing to a "better" prior didn't help a~N(120,20), b~LogN(0,1.5)
# many points away from the mean
# especially at the extreme weights
# consistently overestimates or underestimates true height
# in children, age is an important covariate
# well, to now choose a different model is difficult, because I've looked at the scatter plot



#4.3
height = data['height']
weight = data['weight']

weight = np.log(weight)

weightBar = np.mean(weight)


with pm.Model() as normal_approximation:
	alpha = pm.Normal('alpha', mu=178, sigma=20)
	beta = pm.Lognormal('beta', mu=0, sigma=1)

	s = pm.Uniform('sigma', 0, 20)

	h = pm.Normal('height', mu=alpha + beta*(weight-weightBar), sigma=s, observed=height)

	mean_q = pm.find_MAP()

	hessian = pm.find_hessian(mean_q, vars=[alpha, beta, s])


	covariance_q = np.linalg.inv(hessian)

	std_q = np.sqrt(np.diag(covariance_q))


#draw samples from posterior
samples = np.random.multivariate_normal([mean_q['alpha'], mean_q['beta'], mean_q['sigma']], covariance_q, size=10000)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(np.exp(weight), height, color='w', edgecolor='C0', alpha=0.5)

a_map, b_map, s_map = np.mean(samples,0)

wwe = np.arange(4, 64)
ww = np.log(wwe)
ax.plot(wwe, a_map + b_map * (ww-weightBar), color='k')
mus = samples[:, 0 ] + samples[:, 1] * np.tile(ww.reshape(1,ww.shape[0]) - weightBar, (samples.shape[0],1)).T

ci89 = np.percentile(mus, [1.5, 98.5], axis=1).T


ax.plot(wwe, ci89[:,0], color='k', linestyle='--', alpha=0.7)
ax.plot(wwe, ci89[:,1], color='k', linestyle='--', alpha=0.7)

ax.fill_between(wwe, ci89[:,0], ci89[:,1], color='k', alpha=0.2)

posteriorDraws = np.random.normal(mus, samples[:,2])
posteriorCi89 = np.percentile(posteriorDraws, [1.5, 98.5], axis=1).T

ax.fill_between(wwe, posteriorCi89[:,0], posteriorCi89[:,1], color='k', alpha=0.1)
ax.set_title('4.3')
#pretty good fit?

'''

#4.4
N = 50
alphas = np.random.normal(178, 20, size=N)
beta1s = np.exp(np.random.normal(0, 1, size=N))
beta2s = -np.abs(np.random.normal(0, 2, size=N))
sigmas = 50 * np.random.rand(N)

height = data['height']
weight = data['weight']


weight = np.array(weight)
height = np.array(height)

from sklearn.preprocessing import StandardScaler


ss = StandardScaler()
weightS = ss.fit_transform(weight.reshape(-1,1)).reshape(-1,)

weightS2 = weightS ** 2

xxu = np.linspace(np.min(weight), np.max(weight), 200).reshape(-1,1)
xx = ss.transform(xxu).reshape(-1,)
xx2 = xx ** 2

xx = np.tile(xx, (N,1))
xx2 = np.tile(xx2, (N,1))

mus = (alphas + beta1s * xx.T + beta2s * xx2.T).T

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(N):
	ax.plot(xxu, mus[i], 'k')

# function should be non-decreasing, i.e. 1st derivative positive at least in the range we are interested in
# h(w) = a + β_1 * (w-wbar) + β_2 * (w-wbar)^2
# h'(w) = β_1 + 2 * β_2 * (w-wbar)




plt.show()