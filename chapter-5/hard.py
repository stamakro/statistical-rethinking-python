import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pymc3 as pm

# 5H1
# D independent of M given A, data are consistent

# 5H2
data = pd.read_csv('WaffleDivorce.csv', delimiter=';')

divorce = np.array(data.Divorce)
marriage = np.array(data.Marriage)
ageAtMarriage = np.array(data.MedianAgeMarriage)

divorce = StandardScaler().fit_transform(divorce.reshape(-1,1)).reshape(-1)
marriage = StandardScaler().fit_transform(marriage.reshape(-1,1)).reshape(-1)
ageAtMarriage = StandardScaler().fit_transform(ageAtMarriage.reshape(-1,1)).reshape(-1)

with pm.Model() as normal_approximation:
    alpha = pm.Normal('α_MA', mu=0, sigma=0.2)
    betaM = pm.Normal('β_MA', mu=0, sigma=0.5)

    s = pm.Exponential('σMA', 1)

    alpha1 = pm.Normal('α_AD', mu=0, sigma=0.2)

    betaA = pm.Normal('β_AD', mu=0, sigma=0.5)

    s1 = pm.Exponential('σAD', 1)


    d = pm.Normal('divorce', mu=alpha + betaA * ageAtMarriage, sigma=s1, observed=divorce)

    a = pm.Normal('ageMarriage', mu=alpha + betaM * marriage, sigma=s, observed=ageAtMarriage)

    mean_q = pm.find_MAP()

    hessian = pm.find_hessian(mean_q, vars=[alpha, betaM, s, alpha1, betaA, s1])
    covariance_q = np.linalg.inv(hessian)

    std_q = np.sqrt(np.diag(covariance_q))


samples = np.random.multivariate_normal([mean_q['α_MA'], mean_q['β_MA'], mean_q['σMA'], mean_q['α_AD'], mean_q['β_AD'], mean_q['σAD']], covariance_q, size=10000)

MM = np.linspace(-2.5, 2.5, 60)

aMA = np.tile(samples[:,0], (MM.shape[0], 1)).T
bMA = np.tile(samples[:,1], (MM.shape[0], 1)).T

Am = aMA + bMA * MM

predictedA = np.random.normal(Am, np.tile(samples[:,2], (MM.shape[0], 1)).T)


