import numpy as np, matplotlib.pyplot as plt
from gym import spaces
from matplotlib.pyplot import cm

class delayed_linear_system():
    
    def __init__(self,A,B,C,V,W,Q,R, delay):
    
        self.A, self.B, self.C, self.V, self.W, self.Q, self.R = A, B, C, V, W, Q, R
        self.cholV = np.linalg.cholesky(V)
        self.cholW = np.linalg.cholesky(W)
        self.ydim, self.xdim = C.shape
        self.udim = B.shape[1]
        self.action_space = spaces.Box(-np.inf, +np.inf, (self.udim,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.ydim,), dtype=np.float32)
        
        self.delay = delay
        
    def reset(self, x0 = None):
        
        self.action_buffer = [np.zeros(self.udim)]*self.delay
        self.x = 0.2*np.random.randn(self.xdim) if type(x0)==type(None) else x0
        self.y = self.C@self.x + self.cholW@np.random.randn(self.ydim)
        return self.y
    
    def step(self, action, no_noise = False):
        
        c = 0 if no_noise else 1
        self.action_buffer.append(action)
        u = self.action_buffer.pop(0)
        self.x = self.A@self.x + self.B@u + c*self.cholV@np.random.randn(self.xdim)
        self.y = self.C@self.x + c*self.cholW@np.random.randn(self.ydim)
        self.reward = -(self.x@self.Q*self.x).sum() - (u@self.R*u).sum()
        
        return self.y, self.reward, False, {'u':u}
    
    
    
def visualize_trajectories(env, ABCKL, ep_len = 60, episodes = 3, noisy_env = True):
    
    delay = env.delay
    
    Ah,Bh,Ch,Kh,Lh = ABCKL
    ydim, xdim = Ch.shape
    real_xdim = env.C.shape[1]
    udim = Bh.shape[1]
    
    cost_hist = []
    T = ep_len + 1

    X = np.zeros((T+delay, real_xdim))
    Y = np.zeros((T+delay, ydim))
    U = np.zeros((T+delay-1, udim))

    Xs = np.empty((episodes, T+delay, real_xdim))


    for ep in range(episodes):

        # Setting the iniital condition
        np.random.seed(ep)
        Y[0] = env.reset()
        X[0] = env.x
        Xh = np.zeros((T+delay, xdim))

        Xh[0] = np.linalg.lstsq(Ch,Y[0],rcond=None)[0]

        e = np.zeros((1+delay, ydim))
        ep_cost = 0

        for t in range(delay):

            Xh[t+1] = Ah@Xh[t]

        # timestep iterator
        for t in range(delay, T+delay-1):

            # policy action
            U[t] = -Kh@Xh[t]

            # Delayed time (observations are available after this time)
            t_delay = t - delay + 1

            Y[t_delay], cost, done, info = env.step(U[t], no_noise = not noisy_env)
            X[t_delay] = env.x
            ep_cost += cost

            # updating the e_buffer with new value
            if t_delay >= 0:
                e[1:] = e[:-1]
                e[0] = Y[t_delay] - Ch@Xh[t_delay]

            # Error multiplied by the Kalman gain
            Le = Lh@e[0]

            # Updating the next time step
            Xh[t+1] = Ah@Xh[t] + Bh@U[t] + Le

        for t in range(t+1, t+delay+1):
            t_delay = t - delay + 1
            Y[t_delay], cost, done, info = env.step(0*U[0], no_noise = not noisy_env)
            X[t_delay] = env.x
            ep_cost += cost

        cost_hist.append(ep_cost)

        Xs[ep] = X

    if Xs.shape[-1] ==2 :
        plt.figure(figsize=(14,4));plt.subplot(1,2,1)
        plt.plot(Xs[:,:,0].T,Xs[:,:,1].T);
        [f(0, color='black', linestyle='dashed') for f in [plt.axhline,plt.axvline]];
        plt.xlabel('coord 0', fontsize=14)
        plt.ylabel('coord 1', fontsize=14);
        plt.subplot(1,2,2)
    else:
        plt.figure(figsize=(6,4));
    for i_, trial in enumerate(Xs):
        for coord in range(trial.shape[1]):
            label = 'coord {}'.format(coord) if i_ == 0 else None
            plt.plot(trial[:,coord], color=cm.tab10(coord), label = label)
    plt.legend(fontsize=12)
    plt.xlabel('timestep', fontsize=14)
    plt.ylabel('position', fontsize=14);
