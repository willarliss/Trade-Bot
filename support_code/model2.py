import random
from collections import deque

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

def build_model(input_shape, output_len):

    #h1 = input_shape[0] * input_shape[1] * 3
    #h3 = output_len * 5
    #h2 = int( (h1+h3)/2 )
    
    h1 = 942
    h2 = 360
    h3 = 70
    eta = 0.01
    
    model = Sequential()
    
    model.add( # Input layer
        Flatten(
            input_shape=input_shape,
            )
        )
    
    model.add( # Hidden layer 1
        Dense(
            units=h1, 
            activation='relu',
            )
        )

    model.add( # Hidden layer 2
        Dense(
            units=h2, 
            activation='relu',
            )
        )
    
    model.add( # Hidden layer 3
        Dense(
            units=h3, 
            activation='relu',
            )
        )

    model.add( # Output layer
        Dense(
            units=output_len, 
            activation='linear',
            )
        )
    
    model.compile(
        loss='mse',
        optimizer=Adam(lr=eta),
        )
    
    return model      

class DQN:
    """This class corresponds to testing_DQN-2.ipynb and offers an embedded Double DQN agent"""
    
    def __init__(self, action_space, state_space,
        gamma=0.99, memory_size=100_000, batch_size=100, alpha=1.0, alpha_min=0.01, alpha_decay=0.9,
        ):
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.alpha = alpha
        self.alpha_min = alpha_min
        self.alpha_decay = alpha_decay
        
        self.action_space = action_space
        self.state_space = state_space
        self.memory = deque(maxlen=memory_size)
        
        self.verbose = 0
        self.is_fit = False
        self.model = build_model(state_space, action_space)
        
    def act(self, state):
        
        # Choose random action (i.e. explore environment) depending on exploration rate
        if np.random.rand() < self.alpha:
            action = np.random.randint(self.action_space)
            
        # Choose action based on maximum q-value depending on exploration rate
        else:
            # If nn is fit, predict the q-values (future reward) for each action given the current state
            if self.is_fit:
                q_values = self.model.predict(
                    state.astype(float).reshape(1, *self.state_space)
                    )
            # If nn is not fit, generate random q_values
            else:
                q_values = [np.random.randn(self.action_space)]
            
            # Choose the action based on the greatest 
            action = np.argmax(q_values[0])
            
        return action
    
    def remember(self, state, action, reward, state_next, terminal):
        
        # Commit to memory the current day's state, the action chosen, the reward received for that action,
        # what the next state is, and whether or not the action had a terminal outcome
        self.memory.append(
            (state, action, reward, state_next, terminal)
            )
        
        return self
    
    def replay(self):
        
        # If memory is smaller than batch size, experience replay can't be done
        if len(self.memory) < self.batch_size:
            return 
        
        X, y = [], []
        # Randomly sample a mini batch from memory to use for training
        mini_batch = random.sample(self.memory, self.batch_size)
                
        # Enumerate each object in the mini batch
        for state, action, reward, state_next, terminal in mini_batch:
            
            # If the current state is terminal, the next state's Q-value is just the current reward
            if terminal:
                q_update = reward
            # If the state is not terminal...
            else:
                # If the nn is fit, approximate the next state's Q-value
                if self.is_fit:
                    q_update = reward + self.gamma*np.amax(
                        self.model.predict(state_next.astype(float).reshape(1, *self.state_space))[0]
                        )
                # If the nn is not fit, the next state's Q-value is just the current reward
                else:
                    q_update = reward
            
            # If the nn is fit, approximate the current state's Q-values
            if self.is_fit:
                q_values = self.model.predict(state.astype(float).reshape(1, *self.state_space))
            # If the nn is not fit, set the current state's Q-values to be zeros
            else:
                q_values = np.zeros((1, self.action_space))
            
            # Update the Q-value of the action chosen at the current state to be the Q-value predicted for the next state
            q_values[0][action] = q_update
                    
            # Add the current observation space and the calculated Q_values to their respective arrays for training
            X.append(state)
            y.append(q_values[0])
            
        # Reformat for nn training
        X = np.array(X).reshape(self.batch_size, *self.state_space)
        y = np.array(y).reshape(self.batch_size, self.action_space)
        
        # Fit nn using the chosen observation spaces and their calculated Q-values
        batch = max(8, int(self.batch_size/8))
        self.is_fit = True
        self.model.fit(
            X.astype(float), 
            y.astype(float), 
            batch_size=batch, 
            epochs=50, 
            verbose=self.verbose,
            )

        # Exploration rate decays by set amount
        self.alpha = max(self.alpha_min, self.alpha*self.alpha_decay)
        
        return X, y 
    
    