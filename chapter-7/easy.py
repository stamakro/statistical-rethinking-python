import numpy as np

# 7E1
'''
1) continuous (smoothness)
2) additive
3) increasing in #possible events
'''


# 7E2
e = -0.7*np.log2(0.7) - 0.3 * np.log2(0.3)
print('%.4f' % e)


# 7E3
p = [0.2, 0.25, 0.25, 0.3]
e = 0.

for pp in p:
	e -= pp * np.log2(pp)

print('%.4f' % e)


# 7E4
e = -np.log2(1. / 3.)
print('%.4f' % e)


