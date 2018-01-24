
import numpy as np
import time as T

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.extmath import cartesian

import matplotlib.pyplot as plt


class MPHP:
    '''Multidimensional Periodic Hawkes Process
    Captures rates with periodic component depending on the day of week

    '''

    def __init__(self, alpha=[[0.5]], mu=[0.1], mu_day=np.ones(7), omega=1.0):
        '''params should be of form:
        alpha: numpy.array((u,u)), mu: numpy.array((,u)), omega: float'''

        self.data = []
        self.alpha, self.mu, self.mu_day, self.omega = np.array(alpha), np.array(mu), np.array(mu_day), omega
        self.dim = self.mu.shape[0]
        self.check_stability()

    def check_stability(self):
        ''' check stability of process (max alpha eigenvalue < 1)'''
        w, v = np.linalg.eig(self.alpha)
        me = np.amax(np.abs(w))
        print('Max eigenvalue: %1.5f' % me)
        if me >= 1.:
            print('(WARNING) Unstable.')    

    def generate_seq(self, horizon):
        '''Generate a sequence based on mu, alpha, omega values. 
        Uses Ogata's thinning method, with some speedups, noted below'''

        self.data = []  # clear history
        M = np.sum(self.mu)
        Dstar = np.sum(self.mu_day)
        mu_day_max = np.max(self.mu_day)
        mu_hour_max = np.max(self.mu_hour)

        while True:
            s = np.random.exponential(scale=1. / M)
            day = int(np.floor(s) % 7)
            hour = int(24*(s - day)) 

            # attribute (weighted random sample, since sum(self.mu)==M)
            U = np.random.uniform()
            if U <= self.mu_day[day]/Dstar: 
                event_type = np.random.choice(np.arange(self.dim), 1, p=(self.mu / M)) #[0]
                self.data.append([s, event_type])
                break

        last_rates = self.mu * self.mu_day[day] * self.mu_hour[hour]
        last_day = day
        last_hour = hour
        event_rejected = False

        while True:

            tj, uj = self.data[-1][0], int(self.data[-1][1])
            
            if event_rejected:
                M = np.sum(rates) + np.sum(self.mu) * \
                (mu_day_max*mu_hour_max - self.mu_day[day]*self.mu_hour[hour])
                event_rejected = False

            else: # recalculate M (inclusive of last event)
                M = mu_day_max*mu_hour_max*np.sum(self.mu) + \
                np.sum(last_rates) + self.omega * np.sum(self.alpha[:, uj])

            # generate new event
            s += np.random.exponential(scale=1. / M)
            day = int(np.floor(s) % 7)
            hour = int(24*(s - day)) 

            # calc rates at time s (use trick to take advantage of rates at last event)
            rates = self.mu*self.mu_day[day]*self.mu_hour[hour] +  np.exp(-self.omega * (s - tj)) * \
                (self.alpha[:, uj].flatten() * self.omega + last_rates \
                    - self.mu*self.mu_day[last_day]*self.mu_hour[last_hour])

            # attribution/rejection test
            # handle attribution and thinning in one step as weighted random sample
            diff = M - np.sum(rates)
            
            event_type = np.random.choice(np.arange(self.dim + 1), 1,
                                      p=(np.append(rates, diff) / M))

            if event_type < self.dim:
                self.data.append([s, event_type])
                last_day = day
                last_hour = hour
                last_rates = rates.copy()
            else:
                event_rejected = True

            # if past horizon, done
            if s >= horizon:
                self.data = np.array(self.data)
                self.data = self.data[self.data[:, 0] < horizon]
                return self.data


    def EM_daily(self, Ahat, mhat, mhatday, omega, seq=[], day_reg=np.ones(7), hour_reg=np.ones(24),
     smx=None, tmx=None, regularize=False, Tm=-1, maxiter=100, epsilon=0.01, verbose=True):
        '''implements MAP EM. 
        
        seq[0, :] Time of event in days (float)
        seq[1, :] Event type, indexed 0 to dim-1

        Optional regularization:

        - On excitation matrix Ahat:
         `smx` and `tmx` matrix (shape=(dim,dim)).
        In general, the `tmx` matrix is a pseudocount of parent events from column j,
        and the `smx` matrix is a pseudocount of child events from column j -> i, 
        however, for more details/usage see https://stmorse.github.io/docs/orc-thesis.pdf

        - On day of week parameter mhatday:
        day_reg[i] is a pseudocount of events on the ith day of the week
        Default: day_reg[i] = 1 corresponds to no regularization for ith day
        '''

        # if no sequence passed, uses class instance data
        if len(seq) == 0:
            seq = self.data

        N = len(seq)
        day = (np.floor(seq[:, 0]) % 7).astype(int)
        hour = (24*(s - day)).astype(int)

        self.dim = mhat.shape[0]
        Tm = float(seq[-1, 0]) if Tm < 0 else float(Tm)
        sequ = seq[:, 1].astype(int)

        p_ii = np.random.uniform(0.01, 0.99, size=N)
        p_ij = np.random.uniform(0.01, 0.99, size=(N, N))

        # PRECOMPUTATIONS

        # diffs[i,j] = t_i - t_j for j < i (o.w. zero)
        diffs = pairwise_distances(np.array([seq[:, 0]]).T, metric='euclidean')
        diffs[np.triu_indices(N)] = 0

        # kern[i,j] = omega*np.exp(-omega*diffs[i,j])
        kern = omega * np.exp(-omega * diffs)

        colidx = np.tile(sequ.reshape((1, N)), (N, 1))
        rowidx = np.tile(sequ.reshape((N, 1)), (1, N))

        # approx of Gt sum in a_{uu'} denom **
        seqcnts = np.array([len(np.where(sequ == i)[0]) for i in range(self.dim)])
        seqcnts = np.tile(seqcnts, (self.dim, 1))

        # returns sum of all pmat vals where u_i=a, u_j=b
        # *IF* pmat upper tri set to zero, this is
        # \sum_{u_i=u}\sum_{u_j=u', j<i} p_{ij}
        def sum_pij(a, b):
            c = cartesian([np.where(seq[:, 1] == int(a))[0], np.where(seq[:, 1] == int(b))[0]])
            return np.sum(p_ij[c[:, 0], c[:, 1]])
        vp = np.vectorize(sum_pij)

        # \int_0^t g(t') dt' with g(t)=we^{-wt}
        # def G(t): return 1 - np.exp(-omega * t)
        #   vg = np.vectorize(G)
        # Gdenom = np.array([np.sum(vg(diffs[-1,np.where(seq[:,1]==i)])) for i in range(dim)])

        k = 0
        old_LL = -10000

        while k < maxiter:
            Auu = Ahat[rowidx, colidx] #ahat[i, j] = a_ui, uj
            ag = np.multiply(Auu, kern)
            ag[np.triu_indices(N)] = 0

            # compute m_{u_i}
            self.mu = mhat[sequ]

            # compute delta_{d_i}
            self.mu_day = mhatday[day]
            self.mu_hour = mhathour[hour]

            # compute rates of u_i at time i for all times i 
            rates = self.mu*self.mu_day*self.mu_hour + np.sum(ag, axis=1)

            # compute matrix of p_ii and p_ij  (keep separate for later computations)
            p_ij = np.divide(ag, np.tile(np.array([rates]).T, (1, N)))
            p_ii = np.divide(self.mu, rates)

            # compute mhat:  mhat_u = (\sum_{u_i=u} p_ii) / T
            mhat = np.array([np.sum(p_ii[np.where(seq[:, 1] == i)])
                             for i in range(self.dim)]) / Tm

            mhatday = np.array([np.divide(np.sum(p_ii[np.where(day == i)]) + day_reg[i] - 1, 
                                          np.sum(p_ii)/7 + day_reg[i] - 1) for i in range(7)])

            mhathour = np.array([np.divide(np.sum(p_ii[np.where(hour == i)]) + hour_reg[i] - 1, 
                                          np.sum(p_ii)/24 + hour_reg[i] - 1) for i in range(24)])
            # ahat_{u,u'} = (\sum_{u_i=u}\sum_{u_j=u', j<i} p_ij) / \sum_{u_j=u'} G(T-t_j)
            # approximate with G(T-T_j) = 1
            if regularize:
                Ahat = np.divide(np.fromfunction(lambda i, j: vp(i, j), (self.dim, self.dim)) + (smx - 1),
                                 seqcnts + tmx)
            else:
                Ahat = np.divide(np.fromfunction(lambda i, j: vp(i, j), (self.dim, self.dim)),
                                 seqcnts)

            if k % 10 == 0:
                term1 = np.sum(np.log(rates))
                term2 = Tm * np.sum(mhat)
                term3 = np.sum(np.sum(Ahat[u, int(seq[j, 1])] for j in range(N)) for u in range(self.dim))
                
                new_LL = (1./N) * (term1 - term2 - term3)
                #new_LL = (1. / N) * (term1 - term3)
                
                if abs(new_LL - old_LL) <= epsilon:
                    if verbose:
                        print('Reached stopping criterion. (Old: %1.3f New: %1.3f)' % (old_LL, new_LL))
                        self.alpha = Ahat
                        self.mu = mhat
                        self.mu_day = mhatday
                        self.mu_hour = mhathour
                    return Ahat, mhat, mhatday, mhathour
                if verbose:
                    print('After ITER %d (old: %1.3f new: %1.3f)' % (k, old_LL, new_LL))
                    print(' terms %1.4f, %1.4f, %1.4f' % (term1, term2, term3))

                old_LL = new_LL

            k += 1

        if verbose:
            print('Reached max iter (%d).' % maxiter)

        self.alpha = Ahat
        self.mu = mhat
        self.mu_day = mhatday
        self.mu_hour = mhathour
        return Ahat, mhat, mhatday, mhathour
     

# VISUALIZATION METHODS

    def get_rate(self, ct, d):
        # return rate at time ct in dimension d
        seq = np.array(self.data)
        if not np.all(ct > seq[:, 0]):
            seq = seq[seq[:, 0] < ct]
        return self.mu[d] + \
            np.sum([self.alpha[d, int(j)] * self.omega * np.exp(-self.omega * (ct - t)) for t, j in seq])


    def plot_events(self, horizon=-1, showDays=True, labeled=True):
        if horizon < 0:
            horizon = np.amax(self.data[:, 0])

        fig = plt.figure(figsize=(10, 2))
        ax = plt.gca()
        for i in range(self.dim):
            subseq = self.data[self.data[:, 1] == i][:, 0]
            plt.plot(subseq, np.zeros(len(subseq)) - i, 'bo', alpha=0.2)

        if showDays:
            for j in range(1, int(horizon)):
                plt.plot([j, j], [-self.dim, 1], 'k:', alpha=0.15)

        if labeled:
            ax.set_yticklabels('')
            ax.set_yticks(-np.arange(0, self.dim), minor=True)
            ax.set_yticklabels([r'$e_{%d}$' % i for i in range(self.dim)], minor=True)
        else:
            ax.yaxis.set_visible(False)

        ax.set_xlim([0, horizon])
        ax.set_ylim([-self.dim, 1])
        ax.set_xlabel('Days')
        plt.tight_layout()