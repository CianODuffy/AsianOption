import numpy as np

def fixed_strike_call_payoff(sampling_function, strike):
    strikeArray = np.full(len(sampling_function), strike)
    payoff = np.maximum(sampling_function - strikeArray, 0)
    return payoff

def fixed_strike_put_payoff(sampling_function, strike):
    strikeArray = np.full(len(sampling_function), strike)
    payoff = np.maximum(strikeArray - sampling_function, 0)
    return payoff

def floating_strike_call_payoff(sampling_function, S):
    payoff = np.maximum(S - sampling_function, 0)
    return payoff

def floating_strike_put_payoff(sampling_function, S):
    payoff = np.maximum(sampling_function - S, 0)
    return payoff

def vanilla_put_payoff(S, strike):
    payoff = np.maximum(strike - S, 0)
    return payoff

def vanilla_call_payoff(S, strike):
    payoff = np.maximum(S - strike, 0)
    return payoff

