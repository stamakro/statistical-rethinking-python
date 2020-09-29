import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom



grid = np.linspace(0.0, 1.0, 100)

prior = np.ones(grid.shape[0])
prior /= np.sum(prior)

#W,W,W
likelihood = binom.pmf(3, 3, grid) 

posterior = likelihood * prior

posterior /= np.sum(posterior)



fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(grid, prior, 'k--', label='prior')

ax.plot(grid, posterior, color='C0', label='WWW')



#W,W,W,L
likelihood = binom.pmf(3, 4, grid) 

posterior = likelihood * prior
posterior /= np.sum(posterior)

ax.plot(grid, posterior, color='C1', label='WWWL')

#L,W,W,L,W,W,W
likelihood = binom.pmf(5, 7, grid) 

posterior = likelihood * prior

posterior /= np.sum(posterior)

ax.plot(grid, posterior, color='C2', label='LWWLWWW')

ax.set_title('2M1')


plt.legend()

##---------------------------------------------------------

prior = np.ones(grid.shape[0])
prior[grid<0.5] = 0.0
prior /= np.sum(prior)

#W,W,W
likelihood = binom.pmf(3, 3, grid) 

posterior = likelihood * prior

posterior /= np.sum(posterior)



fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(grid, prior, 'k--', label='prior')

ax.plot(grid, posterior, color='C0', label='WWW')



#W,W,W,L
likelihood = binom.pmf(3, 4, grid) 

posterior = likelihood * prior
posterior /= np.sum(posterior)

ax.plot(grid, posterior, color='C1', label='WWWL')

#L,W,W,L,W,W,W
likelihood = binom.pmf(5, 7, grid) 

posterior = likelihood * prior

posterior /= np.sum(posterior)

ax.plot(grid, posterior, color='C2', label='LWWLWWW')

ax.set_title('2M2')
plt.legend()




plt.show()

