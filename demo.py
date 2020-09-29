import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import pymc3 as pm
from pymc3.plots import forestplot
from sklearn.preprocessing import scale

logging.basicConfig(level=logging.INFO)


def analyze_data(X, y, if_scale=True):
    """
    Function to analyze data
    :param X: input features
    :param y: output
    :param if_scale: if normalize=True, we normalize X and y
    :return: trace, result, yticks
    """
    epa_cols = X.columns.tolist()
    if if_scale:
        X = scale(X, axis=0)
        y = scale(y, axis=0)
    with pm.Model() as Model_Linthipe_SOC:
        alpha = pm.Normal('alpha', mu=0, sd=1)
        beta = pm.Normal('beta', mu=0, sd=1, shape=X.shape[1])
        sigma = pm.InverseGamma('sigma', alpha=2, beta=1)
        y_fit = pm.Normal('y_fit', mu=alpha + pm.math.dot(X, beta), sd=sigma ** (1.0 / 2), observed=y)
    with Model_Linthipe_SOC:
        trace = pm.sample(1000, step=pm.Metropolis(), chains=2)
    result = pm.summary(trace)
    ind = result.index.tolist()
    ind[1:len(epa_cols) + 1] = epa_cols
    result.index = ind
    yticks = ['alpha'] + epa_cols + ["sigma"]
    return trace, result, yticks


if __name__ == "__main__":
    # Please specify the features and observation and input into analyze_data
    # X: features
    # y: observed
    trace, result, yticks = analyze_data(X=features, y=observed)

    # plot results
    ax = forestplot(trace, varnames=['alpha', 'beta', 'sigma'], credible_interval=0.95)[0]
