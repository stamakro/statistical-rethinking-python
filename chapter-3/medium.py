import numpy as np
import pymc3
from scipy.stats import binom, beta

#3M1
grid = np.linspace(0.0, 1.0, 1000)
prior = np.ones(1000)

likelihood = binom.pmf(8, 15, p=grid)

posterior = likelihood * prior
posterior /= np.sum(posterior)

#3M2
np.random.seed(100)
samples = np.random.choice(grid, size=10000, replace=True, p=posterior)

print('2')
print(pymc3.stats.hpd(samples, 0.9))

#3M3
dummyData = binom.rvs(15, samples, size=samples.shape[0])
print('\n3')
print(np.mean(dummyData == 8))


#3M4
dummyData = binom.rvs(9, samples, size=samples.shape[0])
print('\n4')
print(np.mean(dummyData == 6))

#3M5
prior[grid < 0.5] = 0

posterior = likelihood * prior
posterior /= np.sum(posterior)

samples = np.random.choice(grid, size=10000, replace=True, p=posterior)
print('\n5')
print(pymc3.stats.hpd(samples, 0.9))

dummyData = binom.rvs(15, samples, size=samples.shape[0])
print(np.mean(dummyData == 8))


dummyData = binom.rvs(9, samples, size=samples.shape[0])
print(np.mean(dummyData == 6))

#3M6

N = 2500
s = int(np.round(8 * N / 15))
likelihood = binom.pmf(s, N, p=grid)

posterior = likelihood * prior
posterior /= np.sum(posterior)

samples = np.random.choice(grid, size=10000, replace=True, p=posterior)

interval = pymc3.stats.hpd(samples, 0.99)
print('\n6')
print('%d: %f' % (N, interval[1] - interval[0]))


