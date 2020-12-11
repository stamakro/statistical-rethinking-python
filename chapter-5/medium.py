
# 5M1
#C: being criminal (outcome)
#I: income
#A: being an immigrant

# 5M2
#D: #COVID deaths in a city
#P: population of city
#H: #hospitals in city

# 5M3
#More divorce means more single people so easier to get married
#But this also increases age at marriage? (because you get married again at a later age)

# 5M4
import pandas as pd
import numpy as np
import pymc3 as pm
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('WaffleDivorceMormons.csv', delimiter=',')

divorce = np.array(data.Divorce)
marriage = np.array(data.Marriage)
ageAtMarriage = np.array(data.MedianAgeMarriage)
mormons = np.array(data.PercentMormons)
mormons = np.log(mormons)

divorce = StandardScaler().fit_transform(divorce.reshape(-1,1)).reshape(-1)
marriage = StandardScaler().fit_transform(marriage.reshape(-1,1)).reshape(-1)
ageAtMarriage = StandardScaler().fit_transform(ageAtMarriage.reshape(-1,1)).reshape(-1)
mormons = StandardScaler().fit_transform(mormons.reshape(-1,1)).reshape(-1)

with pm.Model() as normal_approximation:
    alpha = pm.Normal('α', mu=0, sigma=0.2)
    betaMa = pm.Normal('βMa', mu=0, sigma=0.5)
    betaA = pm.Normal('βA', mu=0, sigma=0.5)
    #betaMo = pm.Normal('βMo', mu=0, sigma=0.5)

    s = pm.Exponential('σ', 1)

    d = pm.Normal('divorce', mu=alpha + betaMa * marriage + betaA * ageAtMarriage, sigma=s,
                  observed=divorce)

    mean_q = pm.find_MAP()

    hessian = pm.find_hessian(mean_q, vars=[alpha, betaMa, betaA, s])
    covariance_q = np.linalg.inv(hessian)

    std_q = np.sqrt(np.diag(covariance_q))

samples = np.random.multivariate_normal([mean_q['α'], mean_q['βMa'], mean_q['βA'], mean_q['σ']],
                                        covariance_q, 10000)
posteriorIntervals = np.percentile(samples, [5.5, 94.5], axis=0)

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(1,3,1)
x1 = mean_q['α'] + mean_q['βMa'] * marriage + mean_q['βA'] * ageAtMarriage
ax.scatter(divorce, x1)
xx = np.linspace(-2, 2)
ax.plot(xx, xx, 'k--')

ax.set_xlabel('divorce true')
ax.set_ylabel('divorce predicted')
ax.set_title('Without mormon fraction')

with pm.Model() as normal_approximation:
        alpha = pm.Normal('α', mu=0, sigma=0.2)
        betaMa = pm.Normal('βMa', mu=0, sigma=0.5)
        betaA = pm.Normal('βA', mu=0, sigma=0.5)
        betaMo = pm.Normal('βMo', mu=0, sigma=0.5)

        s = pm.Exponential('σ', 1)

        d = pm.Normal('divorce', mu=alpha + betaMa * marriage + betaA * ageAtMarriage + betaMo * mormons , sigma=s, observed=divorce)

        mean_q = pm.find_MAP()

        hessian = pm.find_hessian(mean_q, vars=[alpha, betaMa, betaA, betaMo, s])
        covariance_q = np.linalg.inv(hessian)

        std_q = np.sqrt(np.diag(covariance_q))

samples = np.random.multivariate_normal([mean_q['α'], mean_q['βMa'], mean_q['βA'], mean_q['βMo'], mean_q['σ']], covariance_q, 10000)
posteriorIntervals = np.percentile(samples, [5.5, 94.5], axis=0)



ax = fig.add_subplot(1,3,2)
x2 = mean_q['α'] + mean_q['βMa'] * marriage + mean_q['βA']* ageAtMarriage + mean_q['βMo'] * mormons
ax.scatter(divorce, x2)
xx = np.linspace(-2, 2)
ax.plot(xx, xx, 'k--')

ax.set_xlabel('divorce true')
ax.set_ylabel('divorce predicted')
ax.set_title('Including mormon fraction')

ax = fig.add_subplot(1,3,3)
ax.scatter(x1, x2)
ax.set_xlabel('without')
ax.set_ylabel('with')

#effect is significant, i.e. 89% clearly less than zero, but prediction of only 1 state changes a lot (Utah)
#if you take the log, then 2-3 states change for the better
#we are overfitting now

# 5M5
#Predict obesity from exercise and eating out


