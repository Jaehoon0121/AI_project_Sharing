import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import ship83_3DOF
from ShipModellib import *

def check(p1, p2, base_array):
    idxs = np.indices(base_array.shape) # Create 3D array of indices

    p1 = p1.astype(float)
    p2 = p2.astype(float)

    # Calculate max column idx for each row idx based on interpolated line between two points
    max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) +  p1[1]    
    sign = np.sign(p2[0] - p1[0])
    return idxs[1] * sign <= max_col_idx * sign

def create_polygon(vertices, base, resolution=1, value=1):
    base_array = base  # Initialize your array of zeros

    fill = np.ones(base_array.shape) * True  # Initialize boolean array defining shape fill

    vs = vertices * resolution

    # Create check array for each edge segment, combine into fill array
    for k in range(len(vs)):
        fill = np.all([fill, check(vs[k-1], vs[k], base_array)], axis=0)

    # Set all values inside polygon to 4th input(value, default=1)
    base_array[fill] = value

    return base_array

def create_circle(y, x, r, base, resolution=1, value=1):
    base_array = base

    x1 = x * resolution
    y1 = y * resolution
    r1 = r * resolution

    # Make it more efficient, limit the boundary

    box_lim = [x1 - r1, y1 - r1, x1 + r1, y1 + r1]  

    X, Y = np.ogrid[box_lim[0]:box_lim[2], box_lim[1]:box_lim[3]]  # Make grid for processing
    d = np.sqrt((X - x1) ** 2 + (Y - y1) ** 2)

    c = d <= r1

    t = base_array[box_lim[1]:box_lim[3], box_lim[0]:box_lim[2]]

    t[c] = value
    return base_array


class Environment_naive:
    def __init__(
        self,
        start=[0,0],
        obstacles=[],
        prohibit_area=[],  
        goal = [180,50],
        goal_tolerance=5,
        obs_tolerance=10,
        reward=[0,0,0,0],
        maplimit=[0, 0, 200, 100],
        resolution = 100,
    ):
        self.start = start
        self.x = self.start[0]
        self.y = self.start[1]

        self.obstacles = obstacles
        self.prohibit_area = prohibit_area
        self.goal = goal
        self.goal_tolerance = goal_tolerance
        self.obs_tolerance = obs_tolerance
        self.reward = reward
        self.maplimit = maplimit

        self.isEnd = False
        self.isOut = False
        self.isCollision = False
        self.isGoal = False

        self.resolution = resolution

        self.state = np.array([self.x, self.y], dtype=int)
        
        #  Generate Map Grid (For detail, map expands *resolution times)
        self.map = np.zeros((self.resolution*(maplimit[3] - maplimit[1]), self.resolution*(maplimit[2] - maplimit[0])))
        #  Check whether any value to set inital goal cell in map grid
        if len(self.goal) == 3:
            self.map = create_circle(self.goal[1], self.goal[0], self.goal_tolerance, self.map, self.resolution, self.goal[2])

        #  Change the cell value in map grid for obs. and prohibit area
        for i in self.obstacles:
            self.map = create_circle(i[1], i[0], self.obs_tolerance, self.map, self.resolution)

        for i in self.prohibit_area:
            self.map = create_polygon(i, self.map, self.resolution)

    def reset(self):
        self.x = self.start[0]
        self.y = self.start[1]
        
        self.isEnd = False
        self.isOut = False
        self.isCollision = False
        self.isGoal = False
        
        self.state = np.array([self.x, self.y], dtype=int)

        return self.state

    def get_reward(self):
        x = self.x
        y = self.y

        x_min = self.maplimit[0]
        x_max = self.maplimit[2]
        y_min = self.maplimit[1]
        y_max = self.maplimit[3]

        #  tx, ty -> index of Map table
        tx = int(self.resolution*(x-x_min))
        ty = int(self.resolution*(y-y_min))

        #  Prevent to overflow of index
        if x < x_min or x >= x_max or y < y_min or y >= y_max:
            self.isOut = True
            tx = int(self.resolution*(self.start[0]-x_min))
            ty = int(self.resolution*(self.start[1]-y_min))

        r_location = np.copy(self.goal[:2] - self.state)
        dist_to_goal = np.sqrt(r_location.dot(r_location))

        if dist_to_goal < self.goal_tolerance:
            self.isGoal = True

        if self.map[ty][tx]== 1:
            self.isCollision = True

        if self.isGoal:
            self.isEnd = True
            return self.reward[4]   #1,000
        elif self.isCollision: 
            self.isEnd = True
            return self.reward[3]   #-250
        elif self.isOut:            
            self.isEnd = True
            return self.reward[2]   #-500
        else:
            return self.reward[1]/dist_to_goal   #100/r
        
    def cal_next_state(self, n):
        if n == 0:
            self.x = self.x + 3
        elif n == 1:
            self.y = self.y + 3
        elif n == 2:
            self.y = self.y - 3
        else:
            print('chec the acitno index.')

        self.state = np.array([self.x, self.y], dtype=int)

        reward = self.get_reward() 

        return self.state, reward, self.isEnd


class ActorCritic_naive:
    def __init__(
        self,
        env,
        lr=0.00001,
        gamma=0.98,):
        self.lr = lr
        self.gamma = gamma

        self.env = env

        #  Generate Q, pi table -> shape (map shape, # action)
        self.q = - 10 * np.ones(((self.env.maplimit[3]-self.env.maplimit[1]),(self.env.maplimit[2]-self.env.maplimit[0]),3))
        self.pi = np.random.rand((self.env.maplimit[3]-self.env.maplimit[1]),(self.env.maplimit[2]-self.env.maplimit[0]),3)
  
    def train_net(self, s, a, r, s_prime, done):
        if done:
            d = 0
            s_prime = np.copy(s)  #  Prevent to overflow of index
        else:
            d = 1

        #  When terminal state, target value = only *r
        td_target = r + self.gamma * self.q[s_prime[1], s_prime[0], a] * d
        delta = td_target - self.q[s[1], s[0], a]

        grad = np.log(self.pi[s[1], s[0], a]) * delta

        self.pi[s[1], s[0], a] = self.pi[s[1], s[0], a] + self.lr * grad
        self.q[s[1], s[0], a] = self.q[s[1], s[0], a] + self.lr * grad


class Environment_ship_model:
    def __init__(
        self,
        start=[0,0],
        heading_init=0,
        velocity_init=[0,0,0],
        obstacles=[],
        prohibit_area=[],  
        goal = [180,50],
        goal_tolerance=5,
        obs_tolerance=10,
        reward=[0,0,0,0],
        maplimit=[0, 0, 200, 100],
        resolution = 100,
        dt = 1  
    ):
        self.start = start
        self.x = self.start[0]
        self.y = self.start[1]

        self.ship = np.zeros(3) #(x, y, psi)
        self.velocity_init = velocity_init
        self.velocity = np.array(velocity_init, float)
        self.heading_init = heading_init

        self.obstacles = obstacles
        self.prohibit_area = prohibit_area
        self.goal = goal
        self.goal_tolerance = goal_tolerance
        self.obs_tolerance = obs_tolerance
        self.reward = reward
        self.maplimit = maplimit

        self.isEnd = False
        self.isOut = False
        self.isCollision = False
        self.isGoal = False

        self.dt = dt
        self.resolution = resolution


        #  Ref. https://github.com/cybergalactic/PythonVehicleSimulator
        #  Ref. https://www.fossen.biz/wiley/pythonVehicleSim.php

        #  Generate Ship Model
        self.ship_model = ship83_3DOF.shipClarke83(L=20, B=5, T=3, tau_X=1e5)

        self.state = np.array([self.x, self.y], dtype=int)
        
        self.ship[:2] = np.copy(self.start)
        self.ship[2] = self.heading_init

        #  Generate Map Grid (For detail, map expands *resolution times)
        self.map = np.zeros((self.resolution*(maplimit[3] - maplimit[1]), self.resolution*(maplimit[2] - maplimit[0])))
        #  Check whether any value to set inital goal cell in map grid
        if len(self.goal) == 3:
            self.map = create_circle(self.goal[1], self.goal[0], self.goal_tolerance, self.map, self.resolution, self.goal[2])

        #  Change the cell value in map grid for obs. and prohibit area
        for i in self.obstacles:
            self.map = create_circle(i[1], i[0], self.obs_tolerance, self.map, self.resolution)

        for i in self.prohibit_area:
            self.map = create_polygon(i, self.map, self.resolution)

    def reset(self):
        self.x = self.start[0]
        self.y = self.start[1]

        self.ship = np.zeros(3)
        self.velocity = np.copy(self.velocity_init)
        
        self.isEnd = False
        self.isOut = False
        self.isCollision = False
        self.isGoal = False
        
        self.state = np.array([self.x, self.y], dtype=int)
        self.ship[:2] = np.copy(self.start)
        self.ship[2] = self.heading_init

        return self.state

    def get_reward(self):
        x = self.x
        y = self.y

        x_min = self.maplimit[0]
        x_max = self.maplimit[2]
        y_min = self.maplimit[1]
        y_max = self.maplimit[3]

        #  tx, ty -> index of Map table
        tx = int(self.resolution*(x-x_min))
        ty = int(self.resolution*(y-y_min))

        #  Prevent to overflow of index
        if x < x_min or x >= x_max or y < y_min or y >= y_max:
            self.isOut = True
            tx = int(self.resolution*(self.start[0]-x_min))
            ty = int(self.resolution*(self.start[1]-y_min))

        r_location = np.copy(self.goal[:2] - self.state)
        dist_to_goal = np.sqrt(r_location.dot(r_location))

        if dist_to_goal < self.goal_tolerance:
            self.isGoal = True

        if self.map[ty][tx]== 1:
            self.isCollision = True

        if self.isGoal:
            self.isEnd = True
            return self.reward[4]   #1,000
        elif self.isCollision: 
            self.isEnd = True
            return self.reward[3]   #-250
        elif self.isOut:            
            self.isEnd = True
            return self.reward[2]   #-500
        else:
            return self.reward[1]/dist_to_goal   #100/r
        
    def cal_next_state(self, delta):
        location = np.copy(self.ship)
        velocity = np.copy(self.velocity)
        rudder = delta

        #  Ref. https://github.com/cybergalactic/PythonVehicleSimulator
        #  Ref. https://www.fossen.biz/wiley/pythonVehicleSim.php
        v_prime, l_prime = mainLoop.calculate_3DOF(self.dt, self.ship_model, rudder, location, velocity)

        self.velocity = np.copy(v_prime)
        self.ship = np.copy(l_prime)

        self.x = self.ship[0]
        self.y = self.ship[1]

        self.state = np.copy(self.ship[:2])
        self.state = self.state.astype(int)

        reward = self.get_reward() 

        return self.state, reward, self.isEnd


class ActorCritic_ship_model:
    def __init__(
        self,
        env,
        lr=0.00001,
        gamma=0.98,):
        self.lr = lr
        self.gamma = gamma

        self.env = env

        #  Generate Q, pi table -> shape (map shape, # action)
        self.q = - 10 * np.ones(((self.env.maplimit[3]-self.env.maplimit[1]),(self.env.maplimit[2]-self.env.maplimit[0]),3))
        self.pi = np.random.rand((self.env.maplimit[3]-self.env.maplimit[1]),(self.env.maplimit[2]-self.env.maplimit[0]),3)

    def action(self, n):
        delta = 0
        if n == 0:  #straight
            delta = [0, 'deg']
        elif n == 1:    #left turn
            delta = [20, 'deg']
        elif n == 2:    #right turn
            delta = [-20, 'deg']
        else:
            print("Check the action value.")
        return delta
  
    def train_net(self, s, a, r, s_prime, done):
        if done:
            d = 0
            s_prime = np.copy(s)  #  Prevent to overflow of index
        else:
            d = 1

        #  When terminal state, target value = only *r
        td_target = r + self.gamma * self.q[s_prime[1], s_prime[0], a] * d
        delta = td_target - self.q[s[1], s[0], a]

        grad = np.log(self.pi[s[1], s[0], a]) * delta

        self.pi[s[1], s[0], a] = self.pi[s[1], s[0], a] + self.lr * grad
        self.q[s[1], s[0], a] = self.q[s[1], s[0], a] + self.lr * grad
