#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script is for creating an Environment class. Each environment represents
a bike station with the following methods:
    1) generate: this initializes the bike station with stock characteristics
    2) ping: this communicates with RL Agent with current stock info, reward,
                and episode termination status; iterate to new hour
    3) update: this updates the bike stock based on RL Agent Action
    4) reset: reset all environment properties for new episode training

"""

import numpy as np
import pandas as pd
import json
with open('EXPECTED_BALANCES.json') as json_data:
    expected_balance = json.load(json_data)

class env():
    
    def __init__(self, mode, debug, ID, station_history):
        
        print("Creating A Bike Environment...")
        
        self.mode = mode
        self.seed = np.random.random_integers(0, 10)
        self.num_hours = 23
        self.current_hour = 0
        
        # Set the max and min thresholds
        self.max_threshold = 50
        self.max_target = 35
        self.min_target = 15
        
        if mode == 'actual':
            self.bike_stock_sim = station_history
        else:
            self.bike_stock_sim = self.generate_stock(mode, ID)
            
        self.bike_stock = self.bike_stock_sim.copy() # to be reset to original copy every episode
        self.old_stock = self.bike_stock[0]
        self.new_stock = 0
        self.done = False
        self.reward = 0
        self.bike_moved = 0
        self.debug = debug
        self.ID = str(ID)

        #exp_bike_stock is list of expected balances in next hour
        #Predictions based on Random Forests model
        self.exp_bike_stock_sim = list(np.append(expected_balance[self.ID], None)) 
        self.exp_bike_stock = self.exp_bike_stock_sim.copy()
        self.expected_stock = self.exp_bike_stock[0]
        self.expected_stock_new = 0

        self.actions = [-10, -3, -1, 0]
        self.n_actions = len(self.actions)
        #features of the observation: hour, old stock, new stock
        self.n_features = 1

        self.citibike_df = 0
        self.game_over = False

        
        if self.debug == True:
            print("Generating Bike Stock: {}".format(self.mode))
            print("Bike Stock: {}".format(self.bike_stock))
        
    def generate_stock(self, mode, ID):
        
        # generate a list of 24 hourly bike stock based on mode
        # mode: linear, random, or actual (based on real-world dataset)
        
        bike_stock = [20]
        
        if mode == "linear":
            for i in range(1, 24):
                bike_stock.append(bike_stock[i-1]+3)
                
        if mode == "random":
            for i in range(1, 24):
                bike_stock.append(bike_stock[i-1] + 3 + np.random.random_integers(-5, 5))
                
        if mode == "actual":
            pass
            
        return bike_stock
    
    
    def ping_dqn(self, index):
        
        # share back t+1 stock, reward of t, and termination status
        if self.debug == True:
            print("Current Hour: {}".format(self.current_hour))
            print("Current Stock: {}".format(self.bike_stock[self.current_hour]))
            print("Bikes Moved in Last Hour: {}".format(self.bike_moved))
            print("Collect {} rewards".format(self.reward))
            print("Will move {} bikes".format(action))
            print("---")
        
        action = self.actions[index]
        if action != 0:
            self.update_stock(action)
            self.reward = -0.2*np.abs(action)
            
        if self.bike_stock[self.current_hour] > self.max_target:
            if self.bike_stock[self.current_hour] > self.max_threshold:
                self.reward = -100
            else:
                self.reward = -20
            
        
        if self.bike_stock[self.current_hour] < self.min_target:
            if self.bike_stock[self.current_hour] < 0:
                self.reward = -100
            else:
                self.reward = -20
        
        if self.current_hour == 23:
            if (self.bike_stock[self.current_hour] <= self.max_threshold)&(self.bike_stock[self.current_hour] > 0):
                if (self.bike_stock[self.current_hour] <= self.max_target)&(self.bike_stock[self.current_hour] >= self.min_target):
                    self.reward = 500
                else:
                    self.reward = 100
            
            else: 
                self.reward = -200
                
            self.done = True

        if self.current_hour != 23:
            self.update_hour()
            self.old_stock = self.bike_stock[self.current_hour - 1]
            self.new_stock = self.bike_stock[self.current_hour]
            
        return self.current_hour, self.old_stock, self.new_stock, self.reward, self.done



    def ping(self, action):
        
        # share back t+1 stock, reward of t, and termination status
        if self.debug == True:
            print("Current Hour: {}".format(self.current_hour))
            print("Current Stock: {}".format(self.bike_stock[self.current_hour]))
            print("Bikes Moved in Last Hour: {}".format(self.bike_moved))
            print("Collect {} rewards".format(self.reward))
            print("Will move {} bikes".format(action))
            print("---")
        
        if action != 0:
            self.update_stock(action)
            self.reward = -0.5*np.abs(action)
            
        if self.bike_stock[self.current_hour] > self.max_target:
            if self.bike_stock[self.current_hour] > self.max_threshold:
                self.reward = -100
            else:
                self.reward = -20
            
        if self.bike_stock[self.current_hour] < self.min_target:
            if self.bike_stock[self.current_hour] < 0:
                self.reward = -100
            else:
                self.reward = -20
        
        if self.current_hour == 23:
            if (self.bike_stock[self.current_hour] <= self.max_threshold)&(self.bike_stock[self.current_hour] > 0):
                if (self.bike_stock[self.current_hour] <= self.max_target)&(self.bike_stock[self.current_hour] >= self.min_target):
                    self.reward = 500
                else:
                    self.reward = 100
            
            else: 
                self.reward = -200
            
            self.done = True
            #self.new_stock = 'terminal'
            self.game_over = True

        # update to next hour
        if self.current_hour != 23:
            self.update_hour()
            self.old_stock = self.bike_stock[self.current_hour - 1]
            self.new_stock = self.bike_stock[self.current_hour]
            self.expected_stock = self.exp_bike_stock[self.current_hour - 1]
            if self.current_hour < 23:
                self.expected_stock_new = self.exp_bike_stock[self.current_hour]
            

        return self.current_hour, self.old_stock, self.new_stock, self.expected_stock, self.expected_stock_new, self.reward, self.done, self.game_over
    
    def get_old_stock(self):
        
        return self.old_stock

    def get_expected_stock(self):
        if self.current_hour < 23:
            return self.expected_stock
        else:
            return None
    
    def update_stock(self, num_bike):
        
        # update bike stock based on RL Agent action at t
        if self.current_hour != 23:
            for hour in range(self.current_hour+1, len(self.bike_stock)):
                self.bike_stock[hour] += num_bike
                if hour < len(self.bike_stock)-1:
                    self.exp_bike_stock[hour] += num_bike
                
            self.bike_moved = num_bike
        
        else:
            if self.debug == True:
                print("Last Hour. Cannot Move Bikes.")
            pass
        
        return
    
    def update_hour(self):
        
        # update current_hour 
        self.current_hour += 1
        
        if self.debug == True:
            print("Tick... Forwarded Current Hour")
                
        return
    
    def reset(self):
        
        if self.debug == True:
            print("Reset Environment ...")
        
        self.num_hours = 23
        self.current_hour = 0
        self.bike_stock = self.bike_stock_sim.copy()
        self.exp_bike_stock = self.exp_bike_stock_sim.copy()
        self.done = False
        self.reward = 0
        self.bike_moved = 0
        self.old_stock = self.bike_stock[0]
        self.new_stock = 0
        self.expected_stock = self.exp_bike_stock[0]
        self.expected_stock_new = 0
        #return (self.current_hour, self.old_stock, self.new_stock)
        
    def current_stock(self):
        
        return self.bike_stock[self.current_hour]
    
    def get_sim_stock(self):
        
        return self.bike_stock 
   
