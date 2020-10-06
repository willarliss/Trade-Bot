import gym
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    
    metadata = {'render.modes': ['human']}

    def __init__(self, df, balance_init, actions=9, training=True, train_size=0.8, fee=0.001):
        
        super(TradingEnv, self).__init__()
        assert actions in [5, 7, 9, 11, 21]
        
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df.sort_values('date', inplace=True, ascending=False)
        
        self.balance_init = balance_init
        self.fee = fee
        
        self.verbose = 0
        self.training = training
        self.train_test_split = int(self.df.shape[0]*train_size)
        
        self.action_space = gym.spaces.Discrete(n=actions)
        self._actions = np.linspace(-1, 1, actions)
        
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(6,5))
        
    def _next_observation(self):
                
        self.max_price = np.max(self.df.drop(['date', 'volume'], axis=1).loc[self.current_step:self.first_step-4].values)
        self.max_vol = np.max(self.df.loc[self.current_step:self.first_step-4]['volume'])
        
        obs = self.df.loc[self.current_step: self.current_step-4].copy()
        
        obs[['open', 'high', 'low', 'close']] /= self.max_price
        obs['volume'] /= self.max_vol
        
        obs = obs.drop('date', axis=1).values
        
        obs = np.vstack((obs, [ 
            self.net_worth/self.balance_init, 
            self.balance/self.balance_init,
            self.shares_held,
            self.current_step-self.first_step,
            self.shares_held/self.df.loc[self.current_step, 'close'],
            ]))         

        assert obs.shape == self.observation_space.shape
        return obs
    
    def _take_action_verbose(self, action):
        
        current_price = self.df.loc[self.current_step, 'close']
        action = self._actions[action]
        
        print('current price', current_price)
        print('action', action)
                
        if action > 0:
            total_possible = self.balance / (current_price*(1+self.fee))
            print('total possible to buy', total_possible)
            n_shares_bought = total_possible * action
            print('number bought', n_shares_bought)
                        
            cost = n_shares_bought * current_price
            cost *= (1 + self.fee)
            print('cost of buying', cost)
            
            self.balance -= cost
            self.shares_held += n_shares_bought
            print('balance', self.balance, 'shares held', self.shares_held)
            
        if action < 0:
            total_possible = self.shares_held * 1
            print('total possible to sell', total_possible)
            n_shares_sold = total_possible * -action
            print('number sold', n_shares_sold)
                        
            profit = n_shares_sold * current_price
            profit *= (1 - self.fee)
            print('profit from selling', profit)
            
            self.balance += profit
            self.shares_held -= n_shares_sold
            print('balance', self.balance, 'shares held', self.shares_held)
            
        if action == 0:
            self.balance = self.balance
            self.shares_held = self.shares_held
            print('balance', self.balance, 'shares held', self.shares_held)
            
        self.net_worth_prev = self.net_worth
        self.net_worth = self.balance + (self.shares_held*current_price)
        print('previous net worth', self.net_worth_prev, 'current net worth', self.net_worth)
        
        print()
        
        return self
    
    def _take_action_nonverbose(self, action):
        
        current_price = self.df.loc[self.current_step, 'close']
        action = self._actions[action]                                 

        if action > 0:
            total_possible = self.balance / (current_price*(1+self.fee))
            n_shares_bought = total_possible * action
                        
            cost = n_shares_bought * current_price
            cost *= (1 + self.fee)
            
            self.balance -= cost
            self.shares_held += n_shares_bought
            
        if action < 0:
            total_possible = self.shares_held * 1
            n_shares_sold = total_possible * -action
                        
            profit = n_shares_sold * current_price
            profit *= (1 - self.fee)
            
            self.balance += profit
            self.shares_held -= n_shares_sold
            
        if action == 0:
            self.balance = self.balance
            self.shares_held = self.shares_held
            
        self.net_worth = self.balance + (self.shares_held*current_price)
        
        return self
                
    def _reward_fn(self):

        current_price = self.df.loc[self.current_step, 'close']
        total_possible = self.balance_long / (current_price*(1+self.fee))
        cost = total_possible * current_price
        cost *= (1+self.fee)

        self.balance_long -= cost
        self.shares_held_long += total_possible
        self.net_worth_long = self.balance_long + (self.shares_held_long*current_price)
        
        profit = self.net_worth - self.balance_init
        profit_long = self.net_worth_long - self.balance_init

        return (profit - profit_long) / self.balance_init
    
    def step(self, action):
                
        if self.verbose > 0:
            self._take_action_verbose(action)
        elif self.verbose <= 0:
            self._take_action_nonverbose(action)
        self.current_step += 1
        
        # done
        if self.training:
            done = (self.balance < 0 or self.current_step >= self.train_test_split-1)
        else:
            done = (self.balance < 0 or self.current_step >= self.df.shape[0]-1)
           
        # obs
        obs = self._next_observation()

        # reward
        reward = self._reward_fn()
        
        return obs, reward, done, {}
            
    def reset(self):
        
        if self.training: 
            self.current_step = np.random.randint(5, self.train_test_split-10)
        else: 
            self.current_step = self.train_test_split
        self.first_step = self.current_step
        
        self.net_worth = self.balance_init
        self.balance = self.balance_init
        self.shares_held = 0
        
        self.net_worth_long = self.balance_init
        self.balance_long = self.balance_init
        self.shares_held_long = 0
    
        return self._next_observation()
    
    
    def render(self, mode='human'):
        
        print('Current balance:', self.balance)
        print('Current net worth:', self.net_worth)
        print('Shares held:', self.shares_held)
        print('Profit:', self.net_worth - self.balance_init)
        print('Day range:', self.first_step, self.current_step)
        print('Today:', self.df.loc[self.current_step], sep='\n')
        
    def close(self):
        
        pass
               