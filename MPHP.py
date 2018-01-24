
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

    def generate_seq(self, horizon, last_rates=[], seq=None):
        '''Generate a sequence based on mu, alpha, omega values. 
        Uses Ogata's thinning method, with some speedups, noted below
        
        horizon: time period for which to simulate (in days)
        Start simulation from previous history
            last_rates: list of last rates
            seq: for last event, np.array([time, event type])
        '''

        mu_day_max = np.max(self.mu_day)

        if len(last_rates) == 0:
            seq = [] 
            M = np.sum(self.mu)
            Dstar = np.sum(self.mu_day)

            while True:
                s = np.random.exponential(scale=1. / M)
                day = int(np.floor(s) % 7)

                # attribute (weighted random sample, since sum(self.mu)==M)
                U = np.random.uniform()
                if U <= self.mu_day[day]/Dstar: 
                    event_type = np.random.choice(np.arange(self.dim), 1, p=(self.mu / M))
                    seq.append([s, event_type])
                    break

            last_rates = self.mu * self.mu_day[day]
            last_day = day

        else:
            seq = [tuple(seq)]
            s = seq[0][0]
            horizon = s + horizon
            last_rates = np.array(last_rates)
            last_day = int(np.floor(seq[0][0]) % 7)

        event_rejected = False

        while True:

            tj, uj = seq[-1][0], int(seq[-1][1])

            if event_rejected:
                M = np.sum(rates) + np.sum(self.mu) * (mu_day_max - self.mu_day[day])
                event_rejected = False

            else: # recalculate M (inclusive of last event)
                M = mu_day_max*np.sum(self.mu) + np.sum(last_rates) + self.omega * np.sum(self.alpha[:, uj])

            # generate new event
            s += np.random.exponential(scale=1. / M)
            day = int(np.floor(s) % 7)

            # calc rates at time s (use trick to take advantage of rates at last event)
            rates = self.mu*self.mu_day[day] +  np.exp(-self.omega * (s - tj)) * \
                (self.alpha[:, uj].flatten() * self.omega + last_rates - self.mu*self.mu_day[last_day])

            # attribution/rejection test
            # handle attribution and thinning in one step as weighted random sample
            diff = M - np.sum(rates)

            event_type = np.random.choice(np.arange(self.dim + 1), 1,
                                      p=(np.append(rates, diff) / M))

            if event_type < self.dim:
                seq.append([s, event_type])
                last_day = day
                last_rates = rates.copy()
            else:
                event_rejected = True

            # if past horizon, done
            if s >= horizon:
                if last_rates.tolist():
                    seq.pop(0)
                seq = np.array(seq)
                seq = seq[seq[:, 0] < horizon]

                return seq


    def EM(self, Ahat, mhat, mhatday, omega, seq=[], a=np.ones(7), smx=None, tmx=None, regularize=False,
           Tm=-1, maxiter=100, epsilon=0.01, verbose=True):
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
        a[i] is a pseudocount of events on the ith day of the week
        a[i] = 1 corresponds to no regularization for ith day
        '''

        # if no sequence passed, uses class instance data
        if len(seq) == 0:
            seq = self.data

        N = len(seq)
        day = (np.floor(seq[:, 0]) % 7).astype(int)
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

            # compute rates of u_i at time i for all times i 
            rates = self.mu*self.mu_day + np.sum(ag, axis=1)

            # compute matrix of p_ii and p_ij  (keep separate for later computations)
            p_ij = np.divide(ag, np.tile(np.array([rates]).T, (1, N)))
            p_ii = np.divide(self.mu, rates)

            # compute mhat:  mhat_u = (\sum_{u_i=u} p_ii) / T
            mhat = np.array([np.sum(p_ii[np.where(seq[:, 1] == i)])
                             for i in range(self.dim)]) / Tm

            mhatday = np.array([np.divide(np.sum(p_ii[np.where(day == i)]) + a[i] - 1, 
                                          np.sum(p_ii)/7 + a[i] - 1) for i in range(7)])


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
                    return Ahat, mhat, mhatday
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
        return Ahat, mhat, mhatday
     

    def get_ll(self, omega, ahat, mhat, mhatday, seq = [], Tm = 0):

        if len(seq) == 0:
            seq = self.data
        
        N = len(seq)
        day = (np.floor(seq[:, 0]) % 7).astype(int)
        if Tm==0:
            Tm = np.ceil(seq[-1, 0])
        sequ = seq[:, 1].astype(int)
        dim = mhat.shape[0]
        
        # diffs[i,j] = t_i - t_j for j < i (o.w. zero)
        diffs = pairwise_distances(np.array([seq[:, 0]]).T, metric='euclidean')
        diffs[np.triu_indices(N)] = 0

        # kern[i,j] = omega*np.exp(-omega*diffs[i,j])
        kern = omega * np.exp(-omega * diffs)

        colidx = np.tile(sequ.reshape((1, N)), (N, 1))
        rowidx = np.tile(sequ.reshape((N, 1)), (1, N))

        Auu = ahat[rowidx, colidx] 
        ag = np.multiply(Auu, kern)
        ag[np.triu_indices(N)] = 0

        # compute total rates of u_i at time i
        rates = mhat[sequ]*mhatday[day] + np.sum(ag, axis=1)

        term1 = np.sum(np.log(rates))
        term2 = Tm * np.sum(mhat)
        term3 = np.sum(np.sum(ahat[u, int(seq[j, 1])] for j in range(N)) for u in range(dim))

        loglik = (1./N) * (term1 - term2 - term3)
        return loglik

    def get_rate(self, ct, d, seq=None):
        # return rate at time ct in dimension d
        if not seq:
            seq = np.array(self.data)
        else:
            seq = np.array(seq)
        if not np.all(ct > seq[:, 0]):
            seq = seq[seq[:, 0] < ct]

        day = int(np.floor(ct) % 7)
        return self.mu[d]*self.mu_day[day] + \
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