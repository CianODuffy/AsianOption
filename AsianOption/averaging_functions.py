import numpy as np
import math


#input vector
#returns vector
def arithmetic_rolling_average(data):
    mean = np.zeros_like(data)  # stock matrix
    mean[0] = data[0]  # initial values

    for t in range(1, len(data)):  # stock price paths
        mean[t] = mean[t-1] + (data[t] - mean[t-1])/(t)

    return mean

#returns vector
def geometric_rolling_average_exponential(data):
    mean = np.zeros_like(data)  # stock matrix
    mean[0] = data[0]  # initial values

    for t in range(1, len(data)):  # stock price paths
        mean[t] =((mean[t-1]** t) * data[t])**(1/(t+1))

    return mean

#returns vector
def geometric_rolling_average_logarithm(data):
    mean = np.zeros_like(data)  # stock matrix
    mean[0] = data[0]  # initial values

    for t in range(1, len(data)):  # stock price paths
        mean[t] = math.exp((1/(t+1))* (math.log(data[t]) + t * math.log(mean[t-1])))

    return mean

#input vector
#returns value
def arithmetic_average(data):
    N = len(data)
    mean = np.sum(data, axis=0)/N
    return mean


def geometric_average(data):
    N = len(data)
    logArray = np.log(data)
    mean = np.exp((1 / N) * np.sum(logArray, axis=0))
    return mean

