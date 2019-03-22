from MonteCarloSimulation import *
import MonteCarloWithControlVariate as ControlVariate

#THE PURPOSE OF THIS SCRIPT IS TO CALL THE OTHER METHODS AND PRINT RESULTS
#run main to run the other scripts.

S0 = 100.0  # initial index level
K = 100.0  # strike level
T = 1.0  # call option maturity
r = 0.05  # constant short rate
sigma = 0.2  # constant volatility of diffusion
gbm_dates = pd.DatetimeIndex(start='30-09-2017',
                                 end='30-09-2018',
                                 freq='B')

minPower = 1
maxPower = 6

means = []
stds = []

for power in range(minPower, maxPower, 1):
    I = 10 ** power
    option = asian_option(S0, K, T, r, sigma, I, gbm_dates, len(gbm_dates))
    option.calculate_data()
    means.append(option.get_mean())
    stds.append(option.get_std())


minTimeSteps = 50
maxTimeSteps = 500

for M in range(minTimeSteps, maxTimeSteps, minTimeSteps):
     option = asian_option(S0, K, T, r, sigma, 100000, gbm_dates, M)
     option.calculate_data()
     means.append(option.get_mean())
     stds.append(option.get_std())

# this is to test if close to analytical vanilla call value. It is 0.1 off
option = asian_option(S0, K, T, r, sigma, 100000, gbm_dates, 260)
vanilla_call = option.calculate_vanilla_call_data()
vanilla_put = option.calculate_vanilla_put_data()
#
option = ControlVariate.asian_option(S0, K, T, r, sigma, 100000, gbm_dates, 260)
option.calculate_data()
b = option.calculate_b_optimal()
correlations = option.get_correlations()
em_call_geometric_fixed = option.control_variate(True, 'em_call_geometric_fixed')
em_value = np.exp(-r*T)*np.mean(em_call_geometric_fixed)

controlVariatePayoffs = option.get_all_control_variate_payoffs()

controlVariateValues =  {'em_call_geometric_fixed': np.exp(-r*T)*np.mean(controlVariatePayoffs['em_call_geometric_fixed']),
              'em_put_geometric_fixed': np.exp(-r*T)*np.mean(controlVariatePayoffs['em_put_geometric_fixed']),
              'em_put_arithmetic_fixed': np.exp(-r*T)*np.mean(controlVariatePayoffs['em_put_arithmetic_fixed']),
              'em_put_arithmetic_floating': np.exp(-r*T)*np.mean(controlVariatePayoffs['em_put_arithmetic_floating']),
              'em_call_arithmetic_fixed' : np.exp(-r*T)*np.mean(controlVariatePayoffs['em_call_arithmetic_fixed']),
              'em_call_geometric_floating': np.exp(-r*T)*np.mean(controlVariatePayoffs['em_call_geometric_floating']),
              'em_put_geometric_floating': np.exp(-r*T)*np.mean(controlVariatePayoffs['em_put_geometric_floating']),
              'em_call_arithmetic_floating': np.exp(-r*T)*np.mean(controlVariatePayoffs['em_call_arithmetic_floating'])
              }

controlVariateStd =  {'em_call_geometric_fixed': np.exp(-r*T)*np.std(controlVariatePayoffs['em_call_geometric_fixed']),
              'em_put_geometric_fixed': np.exp(-r*T)*np.std(controlVariatePayoffs['em_put_geometric_fixed']),
              'em_put_arithmetic_fixed': np.exp(-r*T)*np.std(controlVariatePayoffs['em_put_arithmetic_fixed']),
              'em_put_arithmetic_floating': np.exp(-r*T)*np.std(controlVariatePayoffs['em_put_arithmetic_floating']),
              'em_call_arithmetic_fixed' : np.exp(-r*T)*np.std(controlVariatePayoffs['em_call_arithmetic_fixed']),
              'em_call_geometric_floating': np.exp(-r*T)*np.std(controlVariatePayoffs['em_call_geometric_floating']),
              'em_put_geometric_floating': np.exp(-r*T)*np.std(controlVariatePayoffs['em_put_geometric_floating']),
              'em_call_arithmetic_floating': np.exp(-r*T)*np.std(controlVariatePayoffs['em_call_arithmetic_floating'])
              }

np.savetxt('means.csv', means, delimiter=",")
np.savetxt('stds.csv', stds, delimiter=",")


plt.figure(figsize=(10, 6))
plt.plot(means)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(stds)
plt.show()
