import numpy as np
import random as random

class MPP:
    def __init__(self, mu=[0.5], mu_day=np.ones(7)):
        '''params should be of form:
        alpha: numpy.array((u,u)), mu: numpy.array((,u)), omega: float'''

        self.data = []
        self.mu, self.mu_day =  np.array(mu), np.array(mu_day)
        self.dim = self.mu.shape[0]

    def rates(self, seq):

        if len(seq) == 0:
            seq = self.data

        N = len(seq)
        dim = len(np.unique(seq[:,1]))
        Tm = float(seq[-1,0])        
        Ni = np.array([np.sum(seq[:, 1]==i) for i in range(self.dim)])
        rates = Ni/Tm
        day = (np.floor(seq[:, 0]) % 7).astype(int)
        dow = np.array([np.divide(np.sum(day == i), N/7.) for i in range(7)])

        self.mu = rates
        self.mu_day = dow
        self.dim = self.mu.shape[0]

        return rates, dow
    
    def generate_seq(self, window=np.inf, N_events=np.inf):
        '''Generate a sequence based on mu values.
        
        '''
        
        def simulate_window(window, seq, t, mu, mu_day, mu_day_max):

            dim = mu.shape[0]

            for event_type in range(dim):
                while t[event_type] < window:
                    t[event_type] += np.random.exponential(scale=1. / (mu_day_max*mu[event_type]))
                    day = int(np.floor(t[event_type]) % 7)

                    p_accept = mu_day[day] / mu_day_max
                    U = np.random.uniform()

                    if p_accept > U:
                        seq.append([t[event_type], event_type])

            return t, seq   
        
        seq = []
        mu_day_max = float(np.max(self.mu_day))
        t = np.zeros((self.dim,)) 
        
        if N_events < np.inf:
            window = 2*np.ceil(float(N_events)/np.sum(self.mu))
            
            while len(seq) < N_events:
                t, seq = simulate_window(window, seq, t, self.mu, self.mu_day, mu_day_max)
                window = 2*np.ceil(float(N_events - len(seq))/np.sum(self.mu))
            
        else:
            t, seq = simulate_window(window, seq, t, self.mu, self.mu_day, mu_day_max)
        
        seq = np.array(seq)   
        if len(seq) > 0:
            seq = seq[seq[:, 0].argsort()]        
        self.data = seq

        if N_events < np.inf:
            seq = seq[:N_events,]

        return seq   