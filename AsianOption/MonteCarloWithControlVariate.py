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
        self.S_em = self.generate_paths_EM()
        self.em_geometric_means = geometric_average(self.S_em)
        self.em_arithmetic_means = arithmetic_average(self.S_em)

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
            { 'em_call_geometric_fixed': pf.fixed_strike_call_payoff(self.em_geometric_means, self.K),
              'em_put_geometric_fixed': pf.fixed_strike_put_payoff(self.em_geometric_means, self.K),
              'em_put_arithmetic_fixed': pf.fixed_strike_put_payoff(self.em_arithmetic_means, self.K),
              'em_put_arithmetic_floating': pf.floating_strike_put_payoff(self.em_arithmetic_means, self.S_em[-1,:]),
              'em_call_arithmetic_fixed' : pf.fixed_strike_call_payoff(self.em_arithmetic_means, self.K),
              'em_call_geometric_floating': pf.floating_strike_call_payoff(self.em_geometric_means, self.S_em[-1,:]),
              'em_put_geometric_floating': pf.floating_strike_put_payoff(self.em_geometric_means, self.S_em[-1,:]),
              'em_call_arithmetic_floating': pf.floating_strike_call_payoff(self.em_arithmetic_means, self.S_em[-1,:]),
              'vanilla_call_payoff': pf.vanilla_call_payoff(self.S_em[-1, :], self.K),
              'vanilla_put_payoff': pf.vanilla_put_payoff(self.S_em[-1, :], self.K)
              })

    def control_variate(self,isCall, asianOption):

        if(isCall):
            vanillaPrice = np.exp(self.r*self.T)*self.call_value()
            vanillaPayoff = 'vanilla_call_payoff'
        else:
            vanillaPrice = np.exp(self.r*self.T)*self.put_value()
            vanillaPayoff = 'vanilla_put_payoff'

        'em_call_geometric_fixed'

        difference = np.mean(np.subtract(self.payoffs[vanillaPayoff],np.full(self.I, vanillaPrice)))

        controlVariatePayoffs = self.payoffs[asianOption] - \
                                  self.optimalB[asianOption][0, 1] \
                                  * np.subtract(self.payoffs[vanillaPayoff],np.full(self.I, vanillaPrice))

        return  controlVariatePayoffs

    def get_all_control_variate_payoffs(self):
        return { 'em_call_geometric_fixed': self.control_variate(True, 'em_call_geometric_fixed'),
              'em_put_geometric_fixed': self.control_variate(False, 'em_put_geometric_fixed'),
              'em_put_arithmetic_fixed': self.control_variate(False, 'em_put_arithmetic_fixed'),
              'em_put_arithmetic_floating': self.control_variate(False, 'em_put_arithmetic_floating'),
              'em_call_arithmetic_fixed' : self.control_variate(True, 'em_call_arithmetic_fixed'),
              'em_call_geometric_floating': self.control_variate(True, 'em_call_geometric_floating'),
              'em_put_geometric_floating': self.control_variate(False, 'em_put_geometric_floating'),
              'em_call_arithmetic_floating': self.control_variate(True, 'em_call_arithmetic_floating')
              }

    def get_mean(self):
        return np.exp(-self.r*self.T)*np.mean(self.payoffs, axis=0)

    def get_std(self):
        return np.exp(-self.r * self.T) * np.std(self.payoffs, axis=0)

    def calculate_b_optimal(self):
        self.optimalB = {'em_call_geometric_fixed': np.cov(self.payoffs['em_call_geometric_fixed'],self.payoffs['vanilla_call_payoff'])/np.var(self.payoffs['vanilla_call_payoff']),
                         'em_put_geometric_fixed': np.cov(self.payoffs['em_put_geometric_fixed'],self.payoffs['vanilla_put_payoff'])/np.var(self.payoffs['vanilla_put_payoff']),
                         'em_put_arithmetic_fixed': np.cov(self.payoffs['em_put_arithmetic_fixed'],self.payoffs['vanilla_put_payoff'])/np.var(self.payoffs['vanilla_put_payoff']),
                         'em_put_arithmetic_floating': np.cov(self.payoffs['em_put_arithmetic_floating'],self.payoffs['vanilla_put_payoff'])/np.var(self.payoffs['vanilla_put_payoff']),
                         'em_call_arithmetic_fixed' : np.cov(self.payoffs['em_call_arithmetic_fixed'],self.payoffs['vanilla_call_payoff'])/np.var(self.payoffs['vanilla_call_payoff']),
                         'em_call_geometric_floating': np.cov(self.payoffs['em_call_geometric_floating'],self.payoffs['vanilla_call_payoff'])/np.var(self.payoffs['vanilla_call_payoff']),
                         'em_put_geometric_floating': np.cov(self.payoffs['em_put_geometric_floating'],self.payoffs['vanilla_put_payoff'])/np.var(self.payoffs['vanilla_put_payoff']),
                         'em_call_arithmetic_floating':np.cov(self.payoffs['em_call_arithmetic_floating'],self.payoffs['vanilla_call_payoff'])/np.var(self.payoffs['vanilla_call_payoff'])
                         }
        return self.optimalB

    def get_correlations(self):
        correlations = {'em_call_geometric_fixed': np.corrcoef(self.payoffs['em_call_geometric_fixed'],self.payoffs['vanilla_call_payoff']),
                         'em_put_geometric_fixed': np.corrcoef(self.payoffs['em_put_geometric_fixed'],self.payoffs['vanilla_put_payoff']),
                         'em_put_arithmetic_fixed': np.corrcoef(self.payoffs['em_put_arithmetic_fixed'],self.payoffs['vanilla_put_payoff']),
                         'em_put_arithmetic_floating': np.corrcoef(self.payoffs['em_put_arithmetic_floating'],self.payoffs['vanilla_put_payoff']),
                         'em_call_arithmetic_fixed' : np.corrcoef(self.payoffs['em_call_arithmetic_fixed'],self.payoffs['vanilla_call_payoff']),
                         'em_call_geometric_floating': np.corrcoef(self.payoffs['em_call_geometric_floating'],self.payoffs['vanilla_call_payoff']),
                         'em_put_geometric_floating': np.corrcoef(self.payoffs['em_put_geometric_floating'],self.payoffs['vanilla_put_payoff']),
                         'em_call_arithmetic_floating':np.corrcoef(self.payoffs['em_call_arithmetic_floating'],self.payoffs['vanilla_call_payoff'])
                         }
        return correlations

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

