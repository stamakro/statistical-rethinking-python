import numpy as np
import matplotlib.pyplot as plt

# 12M1
productivity = np.array([12, 36, 7, 41])
probs = productivity / np.sum(productivity)

cumulativeProbs = np.cumsum(probs)

clo = np.log(cumulativeProbs / (1-cumulativeProbs))[:-1]


# 12M2
x = np.arange(1, productivity.shape[0] + 1)
fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(x, cumulativeProbs, color='w', edgecolor='k')
ax.bar(x-0.1, cumulativeProbs, width=0.1, color='k', alpha=0.3)
ax.bar(x+0.1, probs, bottom=cumulativeProbs-probs, width=0.1, color='b')

ax.set_xticks(x)

# 12M3
'''
P(Y=0) = p + (1-p) * NB(0;N,pp)
P(Y=y) = (1-p) * NB(y;N,pp)
'''
