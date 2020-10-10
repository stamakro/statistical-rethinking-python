import numpy as np
import pymc3
from scipy.stats import binom

np.random.seed(123123)

boy = 1
girl = 0 

birth1 = np.array([1,0,0,0,1,1,0,1,0,1,0,0,1,1,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,1,0,1,1,1,0,1,0,1,1,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,1,1,1,1,0,1,0,1,1,1,1,1,0,0,1,0,1,1,0,1,0,1,1,1,0,1,1,1,1]) 


birth2 = np.array([0,1,0,1,0,1,1,1,0,0,1,1,1,1,1,0,0,1,1,1,0,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,0,1,1,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,0,1,1,1,0,0,0,0,0,0,1,0,0,0,1,1,0,0,1,0,0,1,1,0,0,0,1,1,1,0,0,0,0])


Nboys = np.sum(birth1) + np.sum(birth2)
Nbirths = birth1.shape[0] + birth2.shape[0]

#3H1
grid = np.linspace(0.0, 1.0, 1000)

prior = np.ones(grid.shape[0])

likelihood = binom.pmf(Nboys, Nbirths, grid)

posterior = prior * likelihood
posterior /= np.sum(posterior)


print('1')
print(grid[np.argmax(posterior)])

#3H2
samples = np.random.choice(grid, size=10000, replace=True, p=posterior)

print('\n2')
print(pymc3.stats.hpd(samples, 0.5))
print(pymc3.stats.hpd(samples, 0.89))
print(pymc3.stats.hpd(samples, 0.97))

#3H3
dummyData = binom.rvs(Nbirths, samples, size=samples.shape[0])
counts = np.array([np.sum(dummyData == i) for i in range(Nbirths+1)])
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(131)

ax.bar(np.arange(Nbirths+1), counts)
ax.axvline(Nboys, color='r')

ax.set_xlabel('# births')
ax.set_ylabel('frequency')
ax.set_title('200 births')


print('\n3\nFit looks very good')


#3H4
dummyData = binom.rvs(100, samples, size=samples.shape[0])
counts = np.array([np.sum(dummyData == i) for i in range(101)])

ax = fig.add_subplot(132)

ax.bar(np.arange(counts.shape[0]), counts)
ax.axvline(np.sum(birth1), color='r')

ax.set_xlabel('# births')
ax.set_ylabel('frequency')
ax.set_title('100 first births')

print('\n4\nObserved #boys smaller than what model expects. Not too much though')

#3H5

Nb = np.sum(birth1 == 0)
dummyData = binom.rvs(Nb, samples, size=samples.shape[0])
counts = np.array([np.sum(dummyData == i) for i in range(Nb+1)])

ax = fig.add_subplot(133)

ax.bar(np.arange(counts.shape[0]), counts)
ax.axvline(np.sum(np.logical_and(birth2, birth1==0)), color='r')

ax.set_xlabel('# births')
ax.set_ylabel('frequency')
ax.set_title('2nd births after girl')

print('\n5\nObserved #boys many more than what model expects. Births are not independent')




plt.show()
