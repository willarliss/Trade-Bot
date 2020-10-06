import os
import pickle

import numpy as np
import pandas as pd
import sqlite3 as sql
from tensorflow.keras.models import load_model
from scipy.stats import pearsonr

def fetch_data(table, db='ticker_data.db'):
    
    connection = sql.connect(db)

    try:

        c = connection.cursor()
        c.execute(f"""
            SELECT * 
            FROM {table};
            """)
        df = pd.DataFrame(c.fetchall(), columns=['date', 'open', 'high', 'low', 'close', 'volume'])

    except Exception as e:
        
        print('EXCEPTION', e)

    connection.close()

    return df

def pickle_model(mod, path='model_info'):
    
    memory = mod.memory
    parameters = np.array([
        mod.action_space, mod.state_space, mod.gamma, 1000, mod.batch_size, mod.alpha, mod.alpha_min, mod.alpha_decay,
        ])
         
    try:
        
        with open(os.path.join(path, 'params.pkl'), 'wb') as f:    
            pickle.dump([memory, parameters], f)

        mod.model.save(os.path.join(path, 'network.pb'))
        
    except FileNotFoundError:
        
        os.mkdir(path)
        
        with open(os.path.join(path, 'params.pkl'), 'wb') as f:    
            pickle.dump([memory, parameters], f)

        mod.model.save(os.path.join(path, 'network.pb'))
        
def unpickle_model(mod_class, path='model_info'):
    
    network = load_model(os.path.join(path, 'network.pb'))
    
    with open(os.path.join(path, 'params.pkl'), 'rb') as f:
        
        info = pickle.load(f)
        memory = info[0]
        parameters = info[1]
    
    model = mod_class(*parameters)
    model.model = network
    
    model.memory = memory
    model.is_fit = True
    
    return model
    
def buy_and_hold(balance_init, back_prices, fee):

    balance = balance_init
    net_worth = balance_init
    shares_held = 0
    net_worth_trace = []
    
    for i in np.arange(len(back_prices)):
        
        net_worth_trace.append(net_worth)
        
        current_price = back_prices[i]
        action = 1.0
        
        total_possible = balance / (current_price*(1+fee))
        n_shares_bought = total_possible * action

        cost = n_shares_bought * current_price
        cost *= (1+fee)

        balance -= cost
        shares_held += n_shares_bought

        net_worth = balance + (shares_held*current_price)
    
    return net_worth_trace

def corr(x, y):
    return np.abs(
        pearsonr(x,y)[0]
        )
    
    