import numpy as np


def standardize(x):
	m = np.mean(x)
	s = np.std(x, ddof=1)

	return (x-m) / s



def precis(samples, names=None):
	# samples: (Nsamples, Nvariables)
	#mean, std, 5.5%, 95.5%
	if names is None:
		names = [str(i) for i in range(samples.shape[1])]

	print('%6s\t%4s\t%5s\t%5s\t%5s' % ('', 'mean', 'std', '5.5%', '94.5%'))
	for s, n in zip(samples.T, names):
		print('%6s\t%1.2f\t%1.3f\t%1.3f\t%1.3f' % (n, np.mean(s), np.std(s, ddof=1), np.percentile(s, 5.5), np.percentile(s, 94.5)))




