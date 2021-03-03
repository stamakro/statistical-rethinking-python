# 7M

## 1
WAIC is more general. For a linear normal model with known variance, large sample size and uniform priors on the weights, they are the same.
[ref](http://www.stat.columbia.edu/~gelman/research/published/waic_understand3.pdf)

## 2
Selection selects best model. Comparison gives a softmax over models.
Selection loses uncertainty, because it does not take into account that two models can be nearly the same.

## 3
A more complicated model requires more observations to be fit properly.

## 4
Narrower prior --> fewer effective parameters.
In the limit, prior is a dirac function so parameter is converted to a constant (i.e. we do not have to estimate it anymore)

## 5
The parameter is less likely to be affected by a single outlier/strange data point

## 6
Too narrow priors remove degrees of freedom the model, meaning that the model might not be flexible enough to describe all the patterns in the data
