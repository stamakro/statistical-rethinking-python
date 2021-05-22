import pandas as pd
import numpy as np
import pymc3 as pm
from bayesian_routines import precis
import arviz as az
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
'''
# 11H1
chimps = pd.read_csv('chimpanzees.csv', delimiter=';')

treatment =  np.array(chimps['prosoc_left'] + 2*chimps['condition'])
treatment1h = np.zeros((treatment.shape[0], 4), int)
for i, t in enumerate(treatment):
	treatment1h[i,t] = 1

actor1h = np.zeros((treatment.shape[0], 7), int)
for i, a in enumerate(chimps['actor']):
	actor1h[i, a-1] = 1


with pm.Model() as m3:
	a = pm.Normal('actor', mu=0, sigma=1.5)
	b = pm.Normal('treatment', mu=0, sigma=0.5, shape=treatment1h.shape[1])

	pp = pm.math.dot(treatment1h, b) + a
	p = pm.invlogit(pp)

	pullLeft = pm.Binomial('left', n=1, p=p, observed=np.array(chimps['pulled_left']))


trace_m3 = pm.sample(1000, chains=4, model=m3)


with pm.Model() as m4:
	a = pm.Normal('actor', mu=0, sigma=1.5, shape=actor1h.shape[1])
	b = pm.Normal('treatment', mu=0, sigma=0.5, shape=treatment1h.shape[1])

	pp = pm.math.dot(treatment1h, b) + pm.math.dot(actor1h, a)
	p = pm.invlogit(pp)

	pullLeft = pm.Binomial('left', n=1, p=p, observed=np.array(chimps['pulled_left']))


trace_m4 = pm.sample(1000, chains=4, model=m4)

comp = az.compare({'m3': trace_m3, 'm4': trace_m4})
print(comp)

# the simpler model (no chimp-specific intercept) is a lot worse (weight~=0
# this is to be expected given the large variations in intercepts (precis output p. 330)
# am I missing something here?


# 11H2
d = pd.read_csv('eagles.csv', delimiter=';')
P = np.array(d['P'] == 'L').astype(int)
A = np.array(d['A'] == 'A').astype(int)
V = np.array(d['V'] == 'L').astype(int)
n = np.array(d['n'])
y = np.array(d['y'])


enc = OneHotEncoder(sparse=False)
P = enc.fit_transform(P.reshape(-1,1))
A = enc.fit_transform(A.reshape(-1,1))
V = enc.fit_transform(V.reshape(-1,1))


with pm.Model() as eagle1:
	a = pm.Normal('intercept', mu=0, sigma=1.5)
	bV = pm.Normal('v', mu=0, sigma=0.5, shape=2)
	bP = pm.Normal('p', mu=0, sigma=0.5, shape=2)
	bA = pm.Normal('a', mu=0, sigma=0.5, shape=2)

	pp = a + pm.math.dot(A, bA) + pm.math.dot(P, bP) + pm.math.dot(V, bV)
	p = pm.invlogit(pp)

	res = pm.Binomial('y', n=np.array(d.n), p=p, observed=y)
	mean_q = pm.find_MAP()
	hessian = pm.find_hessian(mean_q, vars =[a,bV, bP, bA])



covariance_q = np.linalg.inv(hessian)
mean_map = np.hstack((mean_q['intercept'], mean_q['v'], mean_q['a'], mean_q['p']))

samplesMAP = np.random.multivariate_normal(mean_map, covariance_q, 4000)
trace = pm.sample(1000, chains=4, model=eagle1)

precis(samplesMAP)
precis(np.hstack((trace['intercept'].reshape(-1,1), trace['v'], trace['a'], trace['p'])))

ppost = trace['v'].dot(V.T) + trace['p'].dot(P.T) + trace['a'].dot(A.T) + trace['intercept'].reshape(-1,1)

ppost = 1 / (1 + np.exp(-ppost))

p_mm = np.mean(ppost, axis=0)
p_pi = np.percentile(ppost, [5.5, 94.5], axis=0)

ypred = pm.sample_posterior_predictive(trace, model=eagle1)['y']

ypost_m = np.mean(ypred, axis=0)
ypost_ci = np.percentile(ypred, [5.5, 94.5], axis=0)

# interaction model (missing)
with pm.Model() as eagle2:
	a = pm.Normal('intercept', mu=0, sigma=1.5)
	bV = pm.Normal('v', mu=0, sigma=0.5, shape=2)
	bP = pm.Normal('p', mu=0, sigma=0.5, shape=2)
	bA = pm.Normal('a', mu=0, sigma=0.5, shape=2)

	pp = a + pm.math.dot(A, bA) + pm.math.dot(P, bP) + pm.math.dot(V, bV)
	p = pm.invlogit(pp)

	res = pm.Binomial('y', n=np.array(d.n), p=p, observed=y)

# 11H3
data = pd.read_csv('salamanders.csv', delimiter=';')
counts = np.array(data['SALAMAN'])
pct = np.array(data['PCTCOVER'])

ss = StandardScaler()
pct = ss.fit_transform(pct.reshape(-1,1)).reshape(-1,)


age = np.array(data['FORESTAGE'])

age = ss.fit_transform(age.reshape(-1,1)).reshape(-1,)


with pm.Model() as simple:
	a = pm.Normal('a', mu=0., sigma=1.5)
	b = pm.Normal('b_pct', mu=0., sigma=0.5)
	m = a + b * pct
	l = pm.math.exp(m)

	y = pm.Poisson('sal', mu=l, observed=counts)


tprior= pm.sample_prior_predictive(samples=50, model=simple)
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(50):
	aa = tprior['a'][i]
	bb = tprior['b_pct'][i]

	m = np.exp(aa + bb * pct)

	ax.plot(pct, m, alpha=0.2)


mean_quap = pm.find_MAP(model=simple)
hessian_quap = pm.find_hessian(mean_quap, vars=[a,b], model=simple)

covariance_quap = np.linalg.inv(hessian_quap)
mm = np.array([mean_quap['a'], mean_quap['b_pct']])
samples_quap = np.random.multivariate_normal(mm, covariance_quap, 4000)

trace = pm.sample(1000, chains=4, model=simple, tune=1000)


precis(np.hstack((trace['a'].reshape(-1,1), trace['b_pct'].reshape(-1,1))))

predictedCounts = pm.sample_posterior_predictive(trace, model=simple)['sal']

m = np.mean(predictedCounts, axis=0)
ci = np.percentile(predictedCounts, [5.5, 94.5], axis=0)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(pct, counts, color='r', edgecolor='k')

ii = np.argsort(pct)
ax.plot(pct[ii], m[ii], color='k')
ax.fill_between(pct[ii], ci[0, ii], ci[1, ii], color='k', alpha=0.3)


with pm.Model() as mv:
	a = pm.Normal('a', mu=0., sigma=1.5)
	b = pm.Normal('b_pct', mu=0., sigma=0.5)
	c = pm.Normal('b_age', mu=0., sigma=0.5)
	m = a + b * pct + c * age
	l = pm.math.exp(m)

	y = pm.Poisson('sal', mu=l, observed=counts)

trace_mv = pm.sample(1000, chains=4, model=mv, tune=1000)


with pm.Model() as interaction:
	a = pm.Normal('a', mu=0., sigma=1.5)
	b = pm.Normal('b_pct', mu=0., sigma=0.5)
	c = pm.Normal('b_age', mu=0., sigma=0.5)
	d = pm.Normal('b_int', mu=0., sigma=0.5)
	m = a + b * pct + c * age + d * pct * age
	l = pm.math.exp(m)

	y = pm.Poisson('sal', mu=l, observed=counts)



trace_inter = pm.sample(1000, chains=4, model=interaction, tune=1000)

az.compare({'univ': trace, 'multiv': trace_mv, 'inter': trace_inter})




# 11H4
data = pd.read_csv('NWOGrants.csv', delimiter=';')

enc = OneHotEncoder(sparse=False)
disc = enc.fit_transform(np.array(data['discipline']).reshape(-1,1))
gender = enc.fit_transform(np.array(data['gender']).reshape(-1,1))

nn = np.array(data['applications'])
awards = np.array(data['awards'])

with pm.Model() as direct:
	a_d = pm.Normal('d', mu=0., sigma=1.5, shape=disc.shape[1])
	a_g = pm.Normal('g', mu=0., sigma=1.5, shape=2)

	mm = pm.math.dot(disc, a_d) + pm.math.dot(gender, a_g)
	p = pm.invlogit(mm)

	y = pm.Binomial('success', n=nn, p=p, observed=awards)

	trace_direct = pm.sample(1000, chains=4, tune=2000)

with pm.Model() as total:
	a_g = pm.Normal('g', mu=0., sigma=1.5, shape=2)

	mm = pm.math.dot(gender, a_g)
	p = pm.invlogit(mm)

	y = pm.Binomial('success', n=nn, p=p, observed=awards)

	trace_total = pm.sample(1000, chains=4, tune=1000)

# total effect of

# 11H5
# you need to correct for career stage as well to get direct effect
'''

# 11H6
data = pd.read_table('Primates301.csv', delimiter=';')
brain = np.array(data.brain)
social = np.array(data.social_learning)
effort = np.array(data.research_effort)

# stupid but easy to keep all full data
ind = np.where(np.isfinite(brain))
brain = brain[ind]
social = social[ind]
effort = effort[ind]

ind = np.where(np.isfinite(social))
brain = brain[ind]
social = social[ind]
effort = effort[ind]

ind = np.where(np.isfinite(effort))
brain = brain[ind]
social = social[ind]
effort = effort[ind]

brain = np.log(brain)
ss = StandardScaler()
brain = ss.fit_transform(brain.reshape(-1,1)).reshape(-1,)

effort = np.log(effort)
ss = StandardScaler()
effort = ss.fit_transform(effort.reshape(-1,1)).reshape(-1,)

with pm.Model() as model1:
	a = pm.Normal('a', mu=0., sigma=1.5)
	b = pm.Normal('b', mu=0., sigma=0.5)

	l = pm.math.exp(a + b * brain)
	y = pm.Poisson('s', mu=l, observed=social)

	trace = pm.sample(1000, chains=4, tune=1000)

precis(np.hstack((trace['a'].reshape(-1,1), trace['b'].reshape(-1,1))))
pm.traceplot(trace)

# change of 1 std in brain size leads to increase of 2^2.8~=16 occurences of social learning
print('\n\n\n')
with pm.Model() as model2:
	a = pm.Normal('a', mu=0., sigma=1.5)
	b = pm.Normal('b', mu=0., sigma=0.5)
	c = pm.Normal('eff', mu=0., sigma=0.5)

	l = pm.math.exp(a + b * brain + c * effort)
	y = pm.Poisson('s', mu=l, observed=social)

	trace2 = pm.sample(1000, chains=4, tune=1000)

precis(np.hstack((trace2['a'].reshape(-1,1), trace2['b'].reshape(-1,1), trace2['eff'].reshape(-1,1))))
pm.traceplot(trace2)
# the effect of research_effort is considerably higher than that of brain size
# however brain size still has a reliably positive coefficient


# c) DAG
# brain --> Social_true --> Social_observed <-- research_effort
# true social learning is an unobserved confounder
# implied conditional independence: brain _|_ research_effort

with pm.Model() as ci:
	a = pm.Normal('a', mu=0., sigma=1.0)
	b = pm.Normal('b', mu=0., sigma=1.0)
	s = pm.Exponential('s', lam=1.0)

	mu = a + b * brain
	y = pm.Normal('E', mu=mu, sigma=s, observed=effort)

	trace3 = pm.sample(1000, chains=4, tune=1000)

precis(np.hstack((trace3['a'].reshape(-1,1), trace3['b'].reshape(-1,1), trace3['s'].reshape(-1,1))))
# linear model shows that the implied CI is not correct.
# bigger brains are studied more (scientists care more about animals closer to human?)






#plt.show()
