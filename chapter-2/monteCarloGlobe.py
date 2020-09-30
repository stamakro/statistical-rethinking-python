import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, beta

np.random.seed(17021991)
Nsamples = 1000

#posterior distribution of p?
p = np.zeros(Nsamples)
p[0] = 0.5

W = 6
L = 3
N = W+L

for i in range(1, p.shape[0]):
	pnew = np.abs(np.random.normal(p[i-1], 0.1, 1))

	if pnew > 1:
		pnew = 2 - pnew

	q0 = binom.pmf(W, N,p[i-1])
	q1 = binom.pmf(W, N,pnew)

	if np.random.rand() < (q1 / q0):
		p[i] = pnew
	else:
		p[i] = p[i-1]


fig = plt.figure()
ax = fig.add_subplot(111)

ax.hist(p, bins=np.linspace(0, 1,21), density=True, color='C0', edgecolor='k', label='MCMC')
xx = np.linspace(0, 1, 101)

ax.plot(xx, beta.pdf(xx, W+1, L+1), color='C3', label='exact')

plt.legend()

plt.show()
