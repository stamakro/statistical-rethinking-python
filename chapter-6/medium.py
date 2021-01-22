import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def standardize(x):
	m = np.mean(x)
	s = np.std(x, ddof=1)

	return (x-m) / s

# 6M1
'''
4 paths (either through A or B, and then either C->Y or C->V->Y)
Paths going through A need to be closed. We need to condition on A (or C and V, but V is not observed)
'''

np.random.seed(151190)

# 6M2
# X -> Z -> Y

N = 5000

'''
d = np.random.multivariate_normal(np.zeros(3), [[1., corrXZ, 0.],[corrXZ, 1., corrZY],[0., corrZY, 1.0]], N)

X = standardize(d[:,0])
Z = standardize(d[:,1])
Y = standardize(d[:,1])
'''

X = np.random.normal(0, 1, N)
Z = np.random.normal(X, 0.5)
Y = np.random.normal(Z, 1)

#
X = standardize(X)
Z = standardize(Z)
Y = standardize(Y)

print(np.corrcoef(X,Z)[0,1])
print(np.corrcoef(X,Y)[0,1])
print(np.corrcoef(Z,Y)[0,1])




fig = plt.figure()
ax = fig.add_subplot(2,2,1)
ax.scatter(X, Z)
ax.set_xlabel('X')
ax.set_ylabel('Z')


ax = fig.add_subplot(2,2,2)
ax.scatter(Z, Y)
ax.set_xlabel('Z')
ax.set_ylabel('Y')




with pm.Model() as corr:
	a = pm.Normal('a', mu=0, sigma=1.)
	bX = pm.Normal('bX', mu=0, sigma=1.)
	bZ = pm.Normal('bZ', mu=0, sigma=1.)
	s = pm.Exponential('σ', lam=1.0)

	m = bX * X + bZ * Z + a

	y = pm.Normal('Y', mu=m, sigma=s, observed=Y)

	mean_q = pm.find_MAP()

	hessian = pm.find_hessian(mean_q, vars=[a, bX, bZ, s])
	covariance_q = np.linalg.inv(hessian)
	std_q = np.sqrt(np.diag(covariance_q))

samples = np.random.multivariate_normal([mean_q['a'], mean_q['bX'], mean_q['bZ'], mean_q['σ']], covariance_q, size=10000)

print(np.percentile(samples, [4.5, 95.5]))

ax = fig.add_subplot(2,2,3)
f1 = gaussian_kde(samples[:,1])
f2 = gaussian_kde(samples[:,2])
xx= np.linspace(-3,3,200)

ax.plot(xx, f1(xx), color='C0', label='bX')
ax.plot(xx, f2(xx), color='C1', label='bZ')
plt.legend()

ax = fig.add_subplot(2,2,4)
ax.scatter(samples[:,0], samples[:,1])

plt.show()

# if we include an intercept, there's no multicolinearity: bX is around 0 and bZ around 1. But if we omit the intercept, we do find multicolinearity
# If we know Z, also knowing X doesn't add information for Y
# Is this really an argument for not removing X beforehand? If two variables are correlated, but still have non-zero coefficents, that would be a more interesting example
# Or does he mean that you wouldn't know which of the two to remove? It would be wrong to remove Z


# 6M3
'''
top left: Z
top right: no paths entering X, no adjustment
bot left: no adjustment, Z is a collider
bot right: A (path A Z Y)

'''
