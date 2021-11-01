import numpy as np, copy, tqdm
from gym.wrappers.monitor import Monitor

def bio_ofc_train(env_fn, ABCKL = None, delay = None, internal_dim = 2, bias_value = 0, seed=None, max_ep_len=10,
            episodes=1000, sid_lr=1e-3, pi_lr=2e-6, sigma=0.1, beta = 1, momentum=.9995, chckpnt_eps = []):

    # Setting up a max_ep lambda
    ep_len_fcn = (lambda ep:max_ep_len+1) if type(max_ep_len) == int else max_ep_len
    T = ep_len_fcn(episodes) # Setting up the maximum epside length
    
    # setting up lr lambdas
    sid_lr_fn = (lambda ep:sid_lr) if type(sid_lr) == float else sid_lr
    pi_lr_fn = (lambda ep:pi_lr) if type(pi_lr) == float else pi_lr
    
    
    # setting up the environment
    env = env_fn()
    
    # Extracting the environment delay if it is not provided
    if delay == None:
        try: delay = env.delay
        except: raise AttributeError('Please provide the environment delay in case env.delay is not defined.')
    
    # extracting the shape of the environment
    x_dim, y_dim, u_dim = internal_dim, env.observation_space.shape[0], env.action_space.shape[0]
    
    # implementing bias as a constant internal dimension
    bias = (bias_value != 0)
    if bias: x_dim += 1
    
    # initializing weights
    if type(seed) == int: np.random.seed(seed)
        
    if type(ABCKL) == type(None): 
        Ahat, Bhat, Chat, Lhat = [.1*np.random.randn(*a_) for a_ in
                              [[x_dim, x_dim], [x_dim, u_dim], [y_dim, x_dim], [x_dim, y_dim]]]
        Khat = np.zeros((u_dim, x_dim)) 
    else: Ahat, Bhat, Chat, Khat, Lhat = copy.deepcopy(ABCKL)
    K_grad = np.zeros_like(Khat) # Gradient of the controller

    Y = np.zeros((T+delay, y_dim)) # observation state
    ll_hist = [] # log likelihood (observation error)
    cost_hist = [] # cost history
    ep_len_hist = [] # cost history
    ABCKL_hist = {} # weight history
    
    # starting the progress bar
    pbar =  tqdm.tqdm(total=episodes)
    
    # allowing keyboard interrupt
    try:
    
        # episode iterator
        for ep in range(episodes):

            # initializing latent variables and actions
            Xhat = np.zeros((T+delay, x_dim))
            U = np.zeros((T+delay-1, u_dim))

            # Resetting the environment
            Y[0] = env.reset()

            # Get the best Xhat from the past observation.
            try:
                if not bias: Xhat[0] = np.linalg.lstsq(Chat, Y[0], rcond=None)[0]
                else: # with a bias, we separate the Chat into the matrix and the bias part
                    Xhat[0,:-1] = np.linalg.lstsq(Chat[:,:-1], Y[0] - bias_value*Chat[:,-1], rcond=None)[0]
                    Xhat[0,-1] = bias_value
            except:
                print('Ill conditioned lst-sqr solver.')
                break

            # Propagate Xhat from past observation to present
            for t in range(delay):
                Xhat[t+1] = Ahat@Xhat[t]
                if bias: Xhat[t+1,-1] = bias_value

            # initializing starting values
            Z = np.zeros((T+delay,u_dim, x_dim)) # eligibility trace history
            e = np.zeros((1+delay, y_dim)) # error buffer
            ep_cost = [] # tracking episode cost
            ep_ll = [] # tracking episode error

            # timestep iterator
            T_ = ep_len_fcn(ep)
            for t in range(delay, T_+2*delay-1):

                # After T_+delay-1 timesteps, we wait for the env to catch up without updating the internal state
                env_catchup = t>=T_+delay-1

                # policy action if not in catchup mode
                if not env_catchup:

                    xi = sigma*np.random.randn(u_dim)
                    U[t] = -Khat@Xhat[t] + xi

                    # eligibitlity trace
                    Z[t] = beta*Z[t-1] + np.outer(xi, Xhat[t])

                # Delayed time 
                t_delay = t - delay + 1

                # Taking a step and updating the episode cost     
                Y[t_delay], reward, done, info = env.step(U[t]) if not env_catchup else env.step(0*U[0])
                cost = -reward
                ep_cost.append(cost)

                # updating the e_buffer with new values
                if t_delay >= 0:
                    e[1:] = e[:-1]
                    e[0] = Y[t_delay] - Chat@Xhat[t_delay]
                    ep_ll.append((e[0]**2).sum())

                # Error multiplied by the Kalman gain
                Le = Lhat@e[0]

                # Updating the internal state if not in catchup mode
                if not env_catchup: 
                    Xhat[t+1] = Ahat@Xhat[t] + Bhat@U[t] + Le
                    if bias: Xhat[t+1,-1] = bias_value

                # Sys-ID gradient updates
                Ahat += sid_lr_fn(ep)*np.outer(Le, Xhat[t-delay])
                Bhat += sid_lr_fn(ep)*np.outer(Le, U[t-delay])
                Lhat += sid_lr_fn(ep)*np.outer(Le, e[-1])
                Chat += sid_lr_fn(ep)*np.outer(e[0], Xhat[t+1-delay])

                # Controller updates with momentum
                K_grad = momentum*K_grad + cost * Z[t_delay-1]
                Khat += pi_lr_fn(ep) * K_grad

                if done: break


            # Tracking total epsiode cost
            cost_hist.append(np.mean(ep_cost))

            # Tracking MSE between x,y (log-likelihood if the covariances was known)
            ll_hist.append(np.mean(ep_ll))

            # Tracking the episode length hestory
            ep_len_hist.append(t)

            # Tracking the history of the matrices
            if ep in chckpnt_eps:
                ABCKL_hist[ep] = copy.deepcopy([Ahat,Bhat,Chat,Khat,Lhat])

            # progress bar step
            pbar.set_postfix(cost=np.mean(cost_hist[-100:]), sid_err=np.mean(ll_hist[-100:]), ep_len=t, sid_lr='{:.2g}'.format(sid_lr_fn(ep)),pi_lr='{:.2g}'.format(pi_lr_fn(ep)))
            pbar.update(1) 
        
    except KeyboardInterrupt:pass
    except TypeError: print('Type error encountered, learning rate likely too big.')
        
    # closing out the progress bar
    pbar.close()
            
    return (Ahat,Bhat,Chat,Khat,Lhat), (cost_hist, ll_hist, ep_len_hist, ABCKL_hist)



def bio_ofc_eval(env_fn, ABCKL, bias_value, delay = None, seed = 0, ep_len = 100, wind_const = None,
                 record_dir = './saves/video', record = False, record_uid = None, tag = None, render = True):
    
    env = env_fn()
    env.seed(seed)
    
    if delay == None: delay = env.delay
    
    Ah,Bh,Ch,Kh,Lh = copy.deepcopy(ABCKL)
    
    ydim, xdim = Ch.shape
    udim = Bh.shape[1]

    cost_hist = []
    T = ep_len + 1

    Y = np.zeros((T+delay, ydim))
    U = np.zeros((T+delay-1, udim))
    
    bias = (bias_value!=0)
    
    # Setting the iniital condition
    np.random.seed(seed)
    
    if record: env = Monitor(env, record_dir, force = False, resume=True, uid = record_uid)
    
    Y[0] = env.reset()
    
    if wind_const!=None: env.wind = wind_const; Y[0,-1] = wind_const;
        
    if render: env.render(text_ul = tag, text_ur = 't = 0')
    Xh = np.zeros((T+delay, xdim))


    if not bias: Xh[0] = np.linalg.lstsq(Ch, Y[0], rcond=None)[0]
    else: # with a bias, we separate the Ch into the matrix and the bias part
        Xh[0,:-1] = np.linalg.lstsq(Ch[:,:-1], Y[0] - bias_value*Ch[:,-1], rcond=None)[0]
        Xh[0,-1] = bias_value

    e = np.zeros((1+delay, ydim))
    ep_cost = 0

    for t in range(delay):

        Xh[t+1] = Ah@Xh[t]
        if bias: Xh[t+1,-1] = bias_value

    # timestep iterator
    for t in range(delay, T+delay-1):

        # policy action
        U[t] = -Kh@Xh[t]

        # Delayed time (observations are available after this time)
        t_delay = t - delay + 1

        Y[t_delay], reward, done, info = env.step(U[t])
        if wind_const!=None: env.wind = wind_const; Y[t_delay,-1] = wind_const;
        if render: env.render(text_ul = tag, text_ur = 't = {}'.format(t_delay))
        cost_hist.append(-reward)

        if abs(Y[t_delay,0])>1 or abs(Y[t_delay,1])>0.6:
            break

        # updating the e_buffer with new value
        if t_delay >= 0:
            e[1:] = e[:-1]
            e[0] = Y[t_delay] - Ch@Xh[t_delay]

        # Error multiplied by the Kalman gain
        Le = Lh@e[0]

        # Updating the next time step
        Xh[t+1] = Ah@Xh[t] + Bh@U[t] + Le
        if bias: Xh[t+1,-1] = bias_value

    for t in range(t+1, t+delay+1):
        t_delay = t - delay + 1
        Y[t_delay], reward, done, info = env.step(0*U[0])
        cost_hist.append(-reward)
    
    env.close()
    
    return Y, Xh, U, cost_hist, e