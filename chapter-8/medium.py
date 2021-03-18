import numpy as np
import pandas as pd
import pymc3 as pm


# 8M1 & 8M2
'''
There is a triple interaction, we need both water, light and cold


B = C * (a + b * W + c * S + d * W * S)

'''

# 8M3

'''
#ravens depends on amount of pray and amount of pray times amount of wolves



'''

# 8M4
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

with pm.Model() as botany:
	a = pm.Normal('a', mu=0.5, sigma=0.1)

	bw = pm.HalfNormal('b_w', sigma=0.1)
	bs = pm.HalfNormal('b_s', sigma=0.1)

	# interaction term
	bws = pm.HalfNormal('b_ws', sigma=0.1)

	sigma = pm.Exponential('sigma', lam=0.5)

	mu = a + bw * water - bs * shade - bws * water * shade

	outcome = pm.Normal('blooms', mu=mu, sigma=sigma, observed=blooms)
	prior_checks = pm.sample_prior_predictive(samples=50)

	mean_q = pm.find_MAP()
	hessian = pm.find_hessian(mean_q, vars=[a,bw,bs,bws,sigma])


	cov = np.linalg.inv(hessian)
	std = np.sqrt(np.diag(cov))

samples = np.random.multivariate_normal([mean_q['a'], mean_q['b_w'], mean_q['b_s'], mean_q['b_ws'], mean_q['sigma']], cov, size=5000)

# these are horrendous priors
# interaction should also be negative
