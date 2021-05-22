import pandas as pd
import numpy as np
import pymc3 as pm
from bayesian_routines import precis
# 11M1
# the partition function of the two distributions is different


# 11M2
# Change of one unit in the variable causes a change of exp(1.7) in the mean of the outcome

# 11M3/4 p of binomial is in [0,1], lamda of poisson in (0, inf)

# 11M5
# an event that happens less than once per unit time

# 11M6
# binomial: support in {0,1,..,N}, known mean
# poisson:  support in positive integers, known mean

# 11M7
chimps = pd.read_csv('chimpanzees.csv', delimiter=';')

treatment =  np.array(chimps['prosoc_left'] + 2*chimps['condition'])
treatment1h = np.zeros((treatment.shape[0], 4), int)
for i, t in enumerate(treatment):
	treatment1h[i,t] = 1

actor1h = np.zeros((treatment.shape[0], 7), int)
for i, a in enumerate(chimps['actor']):
	actor1h[i, a-1] = 1


with pm.Model() as binomial:
	a = pm.Normal('actor', mu=0, sigma=10., shape=actor1h.shape[1])
	b = pm.Normal('treatment', mu=0, sigma=0.5, shape=treatment1h.shape[1])

	pp = pm.math.dot(treatment1h, b) + pm.math.dot(actor1h, a)
	p = pm.invlogit(pp)

	pullLeft = pm.Binomial('left', n=1, p=p, observed=np.array(chimps['pulled_left']))

	mean_q = pm.find_MAP()
	hessian = pm.find_hessian(mean_q, vars =[a,b])


covariance_q = np.linalg.inv(hessian)
std_q = np.sqrt(np.diag(covariance_q))

mean_map =  np.hstack((mean_q['actor'], mean_q['treatment']))

samplesMAP = np.random.multivariate_normal(mean_map, covariance_q, 4000)

trace = pm.sample(1000, chains=4, model=binomial)

print('QUAP')
precis(samplesMAP)
print('\n\n\nMCMC')
precis(np.hstack((trace['actor'], trace['treatment'])))

# seems quite similar with good priors
# estimate of 2nd chimp becomes very different with wide priors
sys.exit(0)
# 11M8
data = pd.read_csv('Kline.csv', delimiter=';') 
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
population = ss.fit_transform(np.log10(np.array(data['population'])).reshape(-1,1)).reshape(-1,)

contact1h = np.zeros((10, 2))
for i in range(10):
	if data.iloc[i]['contact'] == 'low':
		contact1h[i,0] = 1
	else:
		contact1h[i,1] = 1

with pm.Model() as model:
	a = pm.Normal('intercept', mu=3, sigma=0.5, shape=2)
	b = pm.Normal('coeff', mu=0, sigma=0.2, shape=2)

	ll = pm.math.dot(contact1h, a) + pm.math.dot(contact1h, b)*population

	l = pm.math.exp(ll)

	t = pm.Poisson('tools', mu=l, observed=np.array(data['total_tools']))

	trace = pm.sample(1000, chains=2, cores=2)


data2 = data.drop(9)

ss = StandardScaler()
population = ss.fit_transform(np.log10(np.array(data2['population'])).reshape(-1,1)).reshape(-1,)

contact1h = np.zeros((9, 2))
for i in range(9):
	if data.iloc[i]['contact'] == 'low':
		contact1h[i,0] = 1
	else:
		contact1h[i,1] = 1

with pm.Model() as model:
	a = pm.Normal('intercept', mu=3, sigma=0.5, shape=2)
	b = pm.Normal('coeff', mu=0, sigma=0.2, shape=2)

	ll = pm.math.dot(contact1h, a) + pm.math.dot(contact1h, b)*population

	l = pm.math.exp(ll)

	t = pm.Poisson('tools', mu=l, observed=np.array(data2['total_tools']))

	trace2 = pm.sample(1000, chains=2, cores=2)

# the coefficient for low contact civs is much smaller
# because the low contact civ with most tools was removed from the dataset



