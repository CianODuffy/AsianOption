from gbm_helper import *
from averaging_functions import *
import payoff_functions as pf
from scipy import stats

class asian_option():
    def __init__(self, S0, K, T, r, sigma, I, gbm_dates, M):
        self.S0 = float(S0)
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.I = I
        self.M = M

    def generate_underlying_paths(self):
        self.S_exact = self.generate_paths_exact()
        self.exact_geometric_means = geometric_average(self.S_exact)
        self.exact_arithmetic_means = arithmetic_average(self.S_exact)

        self.S_em = self.generate_paths_EM()
        self.em_geometric_means = geometric_average(self.S_em)
        self.em_arithmetic_means = arithmetic_average(self.S_em)

        # plt.figure(figsize=(10, 6))
        # plt.plot(self.S_exact)
        # plt.show()
        #
        # ph.return_histogram(self.exact_geometric_means)
        # ph.return_histogram(self.exact_arithmetic_means)
        #
        # plt.figure(figsize=(10, 6))
        # plt.plot(self.S_em)
        # plt.show()
        #
        # ph.return_histogram(self.em_geometric_means)
        # ph.return_histogram(self.em_arithmetic_means)



    def generate_paths_exact(self):
        dt = self.T / self.M
        shape = (self.M + 1, self.I)
        S = np.zeros((self.M + 1, self.I), dtype=np.float)
        S[0] = self.S0

        np.random.seed(10000)
        rand = np.random.standard_normal(shape)

        for t in range(1, self.M + 1, 1):
            S[t] = S[t - 1] * np.exp((self.r - self.sigma ** 2 / 2) * dt
                                     + self.sigma * rand[t] * math.sqrt(dt))
        return S

    def generate_paths_EM(self):
        dt = self.T / self.M
        shape = (self.M + 1, self.I)
        S = np.zeros((self.M + 1, self.I), dtype=np.float)
        S[0] = self.S0

        np.random.seed(10000)
        rand = np.random.standard_normal(shape)

        for t in range(1, self.M + 1, 1):
            S[t] = S[t - 1] * (1 + self.r * dt + self.sigma * rand[t] * math.sqrt(dt))
        return S




    def calculate_data(self):
        self.generate_underlying_paths()
        self.payoffs = pd.DataFrame(
            { 'exact_call_geometric_fixed' : pf.fixed_strike_call_payoff(self.exact_geometric_means, self.K),
              'exact_call_arithmetic_fixed': pf.fixed_strike_call_payoff(self.exact_arithmetic_means, self.K),
              'exact_put_arithmetic_fixed': pf.fixed_strike_put_payoff(self.exact_arithmetic_means, self.K),
              'exact_put_arithmetic_floating': pf.floating_strike_put_payoff(self.exact_arithmetic_means, self.S_exact[-1,:]),
              'exact_put_geometric_fixed': pf.fixed_strike_put_payoff(self.exact_geometric_means, self.K),
              'em_call_geometric_fixed': pf.fixed_strike_call_payoff(self.em_geometric_means, self.K),
              'em_put_geometric_fixed': pf.fixed_strike_put_payoff(self.em_geometric_means, self.K),
              'em_put_arithmetic_fixed': pf.fixed_strike_put_payoff(self.em_arithmetic_means, self.K),
              'em_put_arithmetic_floating': pf.floating_strike_put_payoff(self.em_arithmetic_means, self.S_exact[-1,:]),
              'em_call_arithmetic_fixed' : pf.fixed_strike_call_payoff(self.em_arithmetic_means, self.K),
              'exact_call_geometric_floating' : pf.floating_strike_call_payoff(self.exact_geometric_means, self.S_exact[-1,:]),
              'exact_put_geometric_floating': pf.floating_strike_put_payoff(self.exact_geometric_means, self.S_exact[-1,:]),
              'exact_call_arithmetic_floating' : pf.floating_strike_call_payoff(self.exact_arithmetic_means, self.S_exact[-1,:]),
              'em_call_geometric_floating': pf.floating_strike_call_payoff(self.em_geometric_means, self.S_em[-1,:]),
              'em_put_geometric_floating': pf.floating_strike_put_payoff(self.em_geometric_means, self.S_em[-1,:]),
              'em_call_arithmetic_floating': pf.floating_strike_call_payoff(self.em_arithmetic_means, self.S_em[-1,:])
              })

    def get_mean(self):
        return np.exp(-self.r*self.T)*np.mean(self.payoffs, axis=0)

    def get_std(self):
        return np.exp(-self.r * self.T) * np.std(self.payoffs, axis=0)

    def calculate_vanilla_call_data(self):
        self.generate_underlying_paths()
        vanilla_call_em_price = np.exp(-self.r*self.T)*np.mean(pf.vanilla_call_payoff(self.S_em[-1, :], self.K))
        vanilla_call_exact = self.call_value()
        difference = vanilla_call_em_price - vanilla_call_exact

        vanilla_call = {'vanilla_call_em_price': vanilla_call_em_price,
                        'vanilla_call_analytical': vanilla_call_exact,
                        'difference': difference
                        }
        return vanilla_call

    def calculate_vanilla_put_data(self):
        self.generate_underlying_paths()
        vanilla_put_em_price = np.exp(-self.r * self.T) * np.mean(pf.vanilla_put_payoff(self.S_em[-1, :], self.K))
        vanilla_put_exact = self.put_value()
        difference = vanilla_put_em_price - vanilla_put_exact

        vanilla_call = {'vanilla_put_em_price': vanilla_put_em_price,
                        'vanilla_put_analytical': vanilla_put_exact,
                        'difference': difference
                        }
        return vanilla_call

    def d1(self):
        ''' Helper function. '''
        d1 = ((np.log(self.S0 / self.K)
               + (self.r + 0.5 * self.sigma ** 2) * self.T)
              / (self.sigma * np.sqrt(self.T)))
        return d1

    def call_value(self):
        d1 = self.d1()
        d2 = ((np.log(self.S0 / self.K)
               + (self.r - 0.5 * self.sigma ** 2) * self.T)
              / (self.sigma * np.sqrt(self.T)))
        value = (self.S0 * stats.norm.cdf(d1, 0.0, 1.0)
                 - self.K * np.exp(-self.r * self.T) * stats.norm.cdf(d2, 0.0, 1.0))
        return value

    def put_value(self):
        d1 = self.d1()
        d2 = ((np.log(self.S0 / self.K)
               + (self.r - 0.5 * self.sigma ** 2) * self.T)
              / (self.sigma * np.sqrt(self.T)))
        value = (-self.S0 * stats.norm.cdf(-d1, 0.0, 1.0)
                 + self.K * np.exp(-self.r * self.T) * stats.norm.cdf(-d2, 0.0, 1.0))
        return value