class parameters():
    def __init__(self):
        self.network = 'dueling_dqn'    #'dqn, dueling_dqn'
        self.buffer ='baseline' #'priority_replay'
        self.env_seed = 0
        self.n_episodes = 2000
        self.max_t = 1000
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.995
        self.double_dqn = 'enable' #enable, disable
