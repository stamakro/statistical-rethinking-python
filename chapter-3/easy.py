import numpy as np
import pymc3
from scipy.stats import binom, beta

grid = np.linspace(0.0, 1.0, 1000)
prior = np.ones(1000)

likelihood = binom.pmf(6, 9, p=grid)

posterior = likelihood * prior
posterior /= np.sum(posterior)

np.random.seed(100)
samples = np.random.choice(grid, size=10000, replace=True, p=posterior)


print('1')
print(np.mean(samples < 0.2))
print(beta.cdf(0.2, 7, 4))


print('\n2')
print(np.mean(samples > 0.8))
print(beta.sf(0.8, 7, 4))


print('\n3')
print(np.mean(np.logical_and(samples > 0.2, samples < 0.8 )))
print( 1- beta.cdf(0.2, 7, 4) - beta.sf(0.8, 7, 4))

print('\n4')
print(np.percentile(samples, 20))
print(beta.ppf(0.2, 7, 4))

print('\n5')
print(np.percentile(samples, 80))
print(beta.ppf(0.8, 7, 4))

print('\n6')
print(pymc3.stats.hpd(samples, 0.66))

print('\n7')
print(np.percentile(samples, [17, 83]))
print(beta.interval(0.66, 7, 4))

