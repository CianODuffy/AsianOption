import math
import pandas as pd
import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
from IPython import *
warnings.simplefilter('ignore')
from gbm_helper import *
from averaging_functions import *

#this file is to produce plots of stock price and is not called in the actual monte carlo simulation

# model parameters
S0 = 100.0  # initial index level
T = 1.0  # time horizon
r = 0.05  # risk-less short rate
vol = 0.2  # instantaneous volatility

# simulation parameters
np.random.seed(250000)
gbm_dates = pd.DatetimeIndex(start='30-09-2017',
                             end='30-09-2018',
                             freq='B')
M = len(gbm_dates)  # time steps
dt = 1 / 260  # fixed for simplicity
df = math.exp(-r * dt)  # discount factor


#exact solution
def simulate_gbm_exact():
    # stock price paths
    rand = np.random.standard_normal((M, I))  # random numbers
    S = np.zeros_like(rand)  # stock matrix
    S[0] = S0  # initial values
    for t in range(1, M):  # stock price paths
        S[t] = S[t - 1] * np.exp((r - vol ** 2 / 2) * dt
                        + vol * rand[t] * math.sqrt(dt))

    gbm = pd.DataFrame(S[:, 0], index=gbm_dates, columns=['index'])
    gbm['returns'] = np.log(gbm['index'] / gbm['index'].shift(1))

    # Realized Volatility (eg. as defined for variance swaps)
    gbm['rea_var'] = M * np.cumsum(gbm['returns'] ** 2) / np.arange(len(gbm))
    gbm['rea_vol'] = np.sqrt(gbm['rea_var'])
    gbm = gbm.dropna()
    return gbm

#Euler-Marayuma
def simulate_gbm_EM():
    # stock price paths
    rand = np.random.standard_normal((M, I))  # random numbers
    S = np.zeros_like(rand)  # stock matrix
    S[0] = S0  # initial values
    for t in range(1, M):  # stock price paths
        S[t] = S[t - 1] * (1+ r*dt + vol * rand[t] * math.sqrt(dt))

    gbm = pd.DataFrame(S[:, 0], index=gbm_dates, columns=['index'])
    gbm['returns'] = np.log(gbm['index'] / gbm['index'].shift(1))

    # Realized Volatility (eg. as defined for variance swaps)
    gbm['rea_var'] = M * np.cumsum(gbm['returns'] ** 2) / np.arange(len(gbm))
    gbm['rea_vol'] = np.sqrt(gbm['rea_var'])
    gbm = gbm.dropna()
    return gbm


I = 1  # index level paths

exact_gbm = simulate_gbm_exact()
exact_gbm['moving_arithmetic_average'] = arithmetic_rolling_average(exact_gbm['index'])
exact_gbm['geometric_rolling_average_logarithm'] = geometric_rolling_average_logarithm(exact_gbm['index'])
exact_gbm['geometric_rolling_average_exponential'] = geometric_rolling_average_exponential(exact_gbm['index'])

finalArithmeticMean = arithmetic_average(exact_gbm['index'])
checkArithmetic = np.abs(finalArithmeticMean - exact_gbm['moving_arithmetic_average'][-1] < 0.000001)

finalGeometricMean = geometric_average(exact_gbm['index'])
checkGeometric = np.abs(finalGeometricMean - exact_gbm['geometric_rolling_average_logarithm'][-1] < 0.000001)

print_statistics(exact_gbm)
quotes_returns(exact_gbm)
return_histogram(exact_gbm)
return_qqplot(exact_gbm)
realized_volatility(exact_gbm)
rolling_statistics(exact_gbm)

em_gbm = simulate_gbm_EM()
em_gbm['moving_arithmetic_average'] = arithmetic_rolling_average(em_gbm['index'])
em_gbm['geometric_rolling_average_logarithm'] = geometric_rolling_average_logarithm(em_gbm['index'])
em_gbm['geometric_rolling_average_exponential'] = geometric_rolling_average_exponential(em_gbm['index'])

print_statistics(em_gbm)
quotes_returns(em_gbm)
return_histogram(em_gbm)
return_qqplot(em_gbm)
realized_volatility(em_gbm)
rolling_statistics(em_gbm)
