import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from sklearn.metrics import auc

def standardize(grid, p):
	#using stupid rectangular rule, much smarter numerical integration methods exist
	#should also work for grids of variable width

	assert p.shape[0] == grid.shape[0]

	mid = 0.5 * (p[1:] + p[:-1])

	x = grid[1:] - grid[:-1]

	standardizationConstant = np.sum(mid * x)

	#print(auc(grid, p / standardizationConstant))

	return p / standardizationConstant




N = 200
grid = np.linspace(0.0, 1.0, N)
binWidth = grid[1] - grid[0]


prior = np.ones(grid.shape[0])
#prior /= np.sum(prior)
prior = standardize(grid, prior)

#W,W,W
likelihood = binom.pmf(3, 3, grid)

posterior = likelihood * prior
#posterior /= np.sum(posterior)
posterior = standardize(grid, posterior)


fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(grid, prior, 'k--', label='prior')

ax.plot(grid, posterior, color='C0', label='WWW')



#W,W,W,L
likelihood = binom.pmf(3, 4, grid)

posterior = likelihood * prior
#posterior /= np.sum(posterior)
posterior = standardize(grid, posterior)

ax.plot(grid, posterior, color='C1', label='WWWL')

#L,W,W,L,W,W,W
likelihood = binom.pmf(5, 7, grid)

posterior = likelihood * prior

#posterior /= np.sum(posterior)
posterior = standardize(grid, posterior)

ax.plot(grid, posterior, color='C2', label='LWWLWWW')

ax.set_title('2M1')


plt.legend()

##---------------------------------------------------------

prior = np.ones(grid.shape[0])
prior[grid<0.5] = 0.0
#prior /= np.sum(prior)
prior = standardize(grid, prior)

#W,W,W
likelihood = binom.pmf(3, 3, grid)

posterior = likelihood * prior

#posterior /= np.sum(posterior)
posterior = standardize(grid, posterior)


fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(grid, prior, 'k--', label='prior')

ax.plot(grid, posterior, color='C0', label='WWW')



#W,W,W,L
likelihood = binom.pmf(3, 4, grid)

posterior = likelihood * prior
#posterior /= np.sum(posterior)
posterior = standardize(grid, posterior)

ax.plot(grid, posterior, color='C1', label='WWWL')

#L,W,W,L,W,W,W
likelihood = binom.pmf(5, 7, grid)

posterior = likelihood * prior

#posterior /= np.sum(posterior)
posterior = standardize(grid, posterior)

ax.plot(grid, posterior, color='C2', label='LWWLWWW')

ax.set_title('2M2')
plt.legend()







plt.show()
