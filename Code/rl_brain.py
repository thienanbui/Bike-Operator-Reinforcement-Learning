#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script is for creating a RL agent class object. This object has the 
following method:
    
    1) choose_action: this choose an action based on Q(s,a) and greedy eps
    2) learn: this updates the Q(s,a) table
    3) check_if_state_exist: this check if a state exist based on env feedback

"""

import numpy as np
import pandas as pd
from dqn import DeepQNetwork

class agent():
    
    
    def __init__(self, epsilon, lr, gamma, current_stock, debug, expected_stock, model_based, dqn_flag = False, n_features = 1):
        
        print("Created an Agent ...")
        self.actions = [-10, -3, -1, 0]
        self.reward = 0
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma
        self.debug = debug
        self.current_stock = current_stock
        self.expected_stock = expected_stock
        self.model_based = model_based
        self.dqn_flag = dqn_flag
        self.n_features = n_features
        
        # performance metric
        self.q_table = pd.DataFrame(columns = self.actions, dtype = np.float64)
        self.hourly_action_history = []
        self.hourly_stock_history = []
        
        # DQN Parameters
        self.dqn_net = DeepQNetwork(len(self.actions), self.n_features, self.lr, 0.9)
        
       
    def choose_action(self, s, ex):
        
        '''
        This function chooses an action based on Q Table. It also does 
        validation to ensure stock will not be negative after moving bikes.
        Input: 
            - s: current bike stock
            - ex: expected bike stock in subsequent hour (based on random forests prediction)
        
        Output:
            - action: number of bikes to move
        
        '''
        
        self.check_state_exist(s)
        self.current_stock = s
        self.expected_stock = ex
        
        # find valid action based on current stock 
        # cannot pick an action that lead to negative stock
        
        # !!!! remove action validation; only rely on reward/penalty !!!
        # valid_state_action = self.find_valid_action(self.q_table.loc[s, :])
        if self.model_based == True:
            #Take an average of current stock and expected stock
            try:
                avg = int(round(0.5*s + 0.5*ex))
            except:
                avg = s
            self.check_state_exist(avg)
            valid_state_action = self.q_table.loc[avg, :]

        elif self.model_based == False:
            valid_state_action = self.q_table.loc[s, :]
        
        
        if self.dqn_flag:
            
            observation = np.expand_dims(s, axis=0)
            
            if np.random.uniform() < self.epsilon:
                actions_value = self.dqn_net.eval_net(observation).numpy()
                action = np.argmax(actions_value)
            else:
                action = np.random.randint(0, len(self.actions))
        else:
        
            if np.random.uniform() < self.epsilon:

                try:
                    # find the action with the highest expected reward

                    valid_state_action = valid_state_action.reindex(np.random.permutation(valid_state_action.index))
                    action = valid_state_action.idxmax()

                except:
                    # if action list is null, default to 0
                    action = 0

                if self.debug == True:
                    print("Decided to Move: {}".format(action))

            else:

                # randomly choose an action
                # re-pick if the action leads to negative stock
                try:
                    action = np.random.choice(valid_state_action.index)
                except:
                    action = 0

                if self.debug == True:
                    print("Randomly Move: {}".format(action))
        
        self.hourly_action_history.append(action)
        self.hourly_stock_history.append(s)
        
        return action
 
    

    def learn(self, s, a, r, s_, ex, g):

        
        '''
        This function updates Q tables after each interaction with the
        environment.
        Input: 
            - s: current bike stock
            - ex: expected bike stock in next hour
            - a: current action (number of bikes to move)
            - r: reward received from current state
            - s_: new bike stock based on bike moved and new stock
        Output: None
        '''
        
        if self.debug == True:
            print("Moved Bikes: {}".format(a))
            print("Old Bike Stock: {}".format(s))
            print("New Bike Stock: {}".format(s_))
            print("---")
        
        self.check_state_exist(s_)

        if self.model_based == False:
            q_predict = self.q_table.loc[s, a]
        elif self.model_based == True:
            avg = int(round(0.5*s + 0.5*ex))
            self.check_state_exist(avg)
            q_predict = self.q_table.loc[avg, a]
        

        if g == False:
            

            # Updated Q Target Value if it is not end of day  
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        
        else:
            # Update Q Target Value as Immediate reward if end of day
            q_target = r

        if self.model_based == False:
            self.q_table.loc[s, a] += self.lr * (q_target - q_predict)
        elif self.model_based == True:
            self.q_table.loc[avg, a] += self.lr * (q_target - q_predict)
        
        return

    
    def check_state_exist(self, state):
        # If the state does not exist, add it to the Q-table
        if state not in self.q_table.index:
            new_row = pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state)
            self.q_table = pd.concat([self.q_table, new_row.to_frame().T])
    

    def find_valid_action(self, state_action):
        
        '''
        This function check the validity actions in a given state.
        Input: 
            - state_action: the current state under consideration
        Output:
            - state_action: a pandas Series with only the valid actions that
                            will not cause negative stock
        '''
        
        # remove action that will stock to be negative
        
        for action in self.actions:
            if self.current_stock + action < 0:
                
                if self.debug == True:
                    print("Drop action {}, current stock {}".format(action, self.current_stock))
                
                state_action.drop(index = action, inplace = True)
        
        return state_action
        
    
    def print_q_table(self):
        
        print(self.q_table)


    def get_q_table(self):
        
        return self.q_table

    
    def get_hourly_actions(self):
        
        return self.hourly_action_history
    
    def get_hourly_stocks(self):
        
        return self.hourly_stock_history

    
    def reset_hourly_history(self):
        
        self.hourly_action_history = []
        self.hourly_stock_history = []
