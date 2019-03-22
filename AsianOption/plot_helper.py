import math
import numpy as np
import pandas as pd
import scipy.stats as scs
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'
from gbm_helper import *

# taken from Yves Hilpsich lecture notes.

def return_histogram(data):
    ''' Plots a histogram of the returns. '''
    plt.figure(figsize=(10, 6))
    x = np.linspace(min(data), max(data), 100)
    plt.hist(np.array(data), bins=50, normed=True)
    y = dN(x, np.mean(data), np.std(data))
    plt.plot(x, y, linewidth=2)
    plt.xlabel('Final Geometric Average')
    plt.ylabel('frequency/probability')
    plt.grid(True)
    plt.show()