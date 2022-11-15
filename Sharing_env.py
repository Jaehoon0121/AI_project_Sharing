import numpy as np

import ship_con
from ShipModellib import *

def create_circle_rev(y, x, r, base, value=1):
    base_array = base

    x1 = x
    y1 = y
    r1 = r

    box_lim = [x1 - r1, y1 - r1, x1 + r1, y1 + r1]

    X, Y = np.ogrid[box_lim[0]:box_lim[2], box_lim[1]:box_lim[3]]
    d = np.sqrt((X - x1) ** 2 + (Y - y1) ** 2)

    c = d <= r1

    t = base_array[box_lim[1]:box_lim[3], box_lim[0]:box_lim[2]]

    t[c] = value
    return base_array

def pip_test(polygon, point):
  length = len(polygon)-1
  dy2 = point[1] - polygon[0][1]
  intersections = 0
  ii = 0
  jj = 1

  while ii<length:
    dy  = dy2
    dy2 = point[1] - polygon[jj][1]

    # consider only lines which are not completely above/bellow/right from the point
    if dy*dy2 <= 0.0 and (point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]):
        
      # non-horizontal line
      if dy<0 or dy2<0:
        F = dy*(polygon[jj][0] - polygon[ii][0])/(dy-dy2) + polygon[ii][0]

        if point[0] > F: # if line is left from the point - the ray moving towards left, will intersect it
          intersections += 1
        elif point[0] == F: # point on line
          return 2

      # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
      elif dy2==0 and (point[0]==polygon[jj][0] or (dy==0 and (point[0]-polygon[ii][0])*(point[0]-polygon[jj][0])<=0)):
        return 2

      # there is another posibility: (dy=0 and dy2>0) or (dy>0 and dy2=0). It is skipped 
      # deliberately to prevent break-points intersections to be counted twice.
    
    ii = jj
    jj += 1
            
  #print 'intersections =', intersections
  return intersections & 1  

def tiling_rev(polygon, base):
    temp_base = base
    temp_poly = polygon

    x_min = int(temp_poly[:,0].min())
    x_max = int(temp_poly[:,0].max())+1
    y_min = int(temp_poly[:,1].min())
    y_max = int(temp_poly[:,1].max())+1
    
    for i in range(x_min, x_max):
        for j in range(y_min, y_max):
            temp_point = [i, j]

            temp_base[j, i] = pip_test(temp_poly, temp_point)

    return temp_base


class Environment_con:
    def __init__(
        self,
        start=np.array([0,0]),
        heading_init=0,
        velocity_init=[0,0,0],
        obstacles=np.array([]),
        prohibit_area=np.array([]),  
        goal=np.array([180,50]),
        goal_tolerance=5,
        obs_tolerance=10,
        reward=[0,0,0,0],
        maplimit=np.array([0, 0, 200, 100]),
        resolution = 0.1,
        dt = 1  
    ):        
        self.dt = dt
        self.resolution = resolution

        self.start = start * self.resolution
        self.x = self.start[0]
        self.y = self.start[1]

        self.ship = np.zeros(3) #(x, y, psi)
        self.velocity_init = velocity_init
        self.velocity = np.array(velocity_init, float)
        self.heading_init = heading_init
        self.rudder = [0]

        self.obstacles = obstacles
        self.prohibit_area = prohibit_area * self.resolution
        self.goal = goal * self.resolution
        self.goal_tolerance = goal_tolerance
        self.obs_tolerance = obs_tolerance
        self.reward = reward
        self.maplimit = maplimit * self.resolution

        self.x_min = self.maplimit[0]
        self.x_max = self.maplimit[2]
        self.y_min = self.maplimit[1]
        self.y_max = self.maplimit[3]

        self.isEnd = False
        self.isOut = False
        self.isCollision = False
        self.isGoal = False

        self.ship_model = ship_con.ship_con(L=20, B=5, T=3, tau_X=1e5)

        self.state = np.array([int(self.x), int(self.y)], dtype=int)
        
        self.ship[:2] = np.copy(self.start)
        self.ship[2] = self.heading_init

        self.shape = np.array(((self.y_max - self.y_min), (self.x_max - self.x_min)))
        self.shape = self.shape.astype(int)
        self.map = np.zeros(self.shape)
 
        for i in self.obstacles:
            self.map = create_circle_rev(i[1], i[0], self.obs_tolerance, self.map)

        for i in self.prohibit_area:
            self.map = tiling_rev(i, self.map)

        if len(self.goal) == 3:
            self.map = create_circle_rev(self.goal[1], self.goal[0], self.goal_tolerance, self.map, self.goal[2])

        temp = np.subtract(self.goal, self.start)
        self.distance_list = np.array([np.sqrt(temp.dot(temp))])
        self.dif_distance_list = np.array([0])

    def reset(self):
        self.x = self.start[0]
        self.y = self.start[1]

        self.ship = np.zeros(3)
        self.velocity = np.copy(self.velocity_init)
        
        self.isEnd = False
        self.isOut = False
        self.isCollision = False
        self.isGoal = False
        
        self.state = np.array([int(self.x), int(self.y)], dtype=int)
        self.ship[:2] = np.copy(self.start)
        self.ship[2] = self.heading_init

        self.rudder = [0]

        return self.state

    def get_reward(self, t = 30):
        x = self.x
        y = self.y

        tx = int(x-self.x_min)
        ty = int(y-self.y_min)

        if x < self.x_min or y < self.y_min or x >= int(self.x_max - 1)  or y >= int(self.y_max - 1):
            self.isOut = True
            tx = int(self.start[0]-self.x_min)
            ty = int(self.start[1]-self.y_min)

        r_location = np.copy(self.goal[:2] - self.state)
        dist_to_goal = np.sqrt(r_location.dot(r_location))

        near_map = self.map[ty - t : ty + t, tx - t : tx + t]
        near_obs = False

        if np.sum(near_map) > 0:
            X, Y = np.ogrid[tx - t : tx + t, ty - t : ty + t]
            d = np.sqrt((X - tx) ** 2 + (Y - ty) ** 2)

            c = d <= t

            temp = c * near_map

            if np.sum(temp) > 0:
                near_obs = True

        self.distance_list = np.append(self.distance_list, dist_to_goal)
        self.dif_distance_list = np.append(self.dif_distance_list, self.distance_list[-2] - dist_to_goal)

        if dist_to_goal < self.goal_tolerance:
            self.isGoal = True

        if self.map[ty][tx]== 1:
            self.isCollision = True

        if self.isGoal:
            self.isEnd = True
            return self.reward[0]   #1,000
        elif self.isOut:            
            self.isEnd = True
            return self.reward[1]   #-500
        elif self.isCollision: 
            self.isEnd = True
            return self.reward[2]   #-350
        elif near_obs:
            return self.reward[3]
        else:
            if dist_to_goal > self.distance_list[-2]:
                return self.reward[4]
            else:
                if self.dif_distance_list[-1] > np.mean(self.dif_distance_list[:-1]):
                    return self.reward[5]
                else:
                    return self.reward[6]
        
    def cal_next_state(self, delta):
        location = np.copy(self.ship)
        velocity = np.copy(self.velocity)
        uc = delta

        v_prime, l_prime, self.rudder = mainLoop.calculate_3DOF_con(self.dt, self.ship_model, self.rudder, uc, location, velocity)

        self.velocity = np.copy(v_prime)
        self.ship = np.copy(l_prime)

        self.x = self.ship[0]
        self.y = self.ship[1]

        self.state = np.copy(self.ship[:2])
        self.state = self.state
        self.state = self.state.astype(int)

        reward = self.get_reward() 

        return self.state, reward, self.isEnd


class ActorCritic_con:
    def __init__(   
        self,
        env,
        lr=0.00001,
        gamma=0.98,
        dirichlet=True,
        action_list = [[131550, 0, 'deg'],[85660, 20, 'deg'],[85660, -20, 'deg']]):
        self.lr = lr
        self.gamma = gamma

        self.env = env
        self.resolution = self.env.resolution
        self.v = np.zeros(self.env.shape)

        self.dirichlet = dirichlet
        if self.dirichlet:
            self.pi = np.random.dirichlet((1, 1, 1), self.env.shape)
        else:
            self.pi = np.random.rand(self.env.shape[0],self.env.shape[1],3)

        self.action_list = action_list

    def action(self, n):
        if n == 0:  #straight
            delta = [131550, 0, 'deg']
        elif n == 1:    #left turning max
            delta = [85660, 20, 'deg']
        elif n == 2:    #right turning max
            delta = [85660, -20, 'deg']
        else:
            print("Check the action value.")

        return delta
  
    def train_net(self, s, a, r, s_prime, done):
        if done:
            d = 0
            s_prime = np.copy(s) 
        else:
            d = 1
        target = r + self.gamma * self.v[s_prime[1], s_prime[0]] * d
        delta = target - self.v[s[1], s[0]]


        pi = self.pi[s[1]:s_prime[1]+1, s[0]:s_prime[0]+1, a]
        pi[np.where(pi <= 0)] = 1e-6

        grad = np.log(pi) * delta

        grad[np.where(grad < -1e3)] = -1e3
        grad[np.where(grad > 1e3)] = 1e3

        if self.dirichlet:
            self.v[s[1]:s_prime[1]+1, s[0]:s_prime[0]+1] = self.v[s[1]:s_prime[1]+1, s[0]:s_prime[0]+1] + self.lr * grad
            self.pi[s[1]:s_prime[1]+1, s[0]:s_prime[0]+1, a] = self.pi[s[1]:s_prime[1]+1, s[0]:s_prime[0]+1, a] + self.lr * grad
            temp1 = self.pi[s[1]:s_prime[1]+1, s[0]:s_prime[0]+1, a-1]
            temp2 = self.pi[s[1]:s_prime[1]+1, s[0]:s_prime[0]+1, a-2]
            self.pi[s[1]:s_prime[1]+1, s[0]:s_prime[0]+1, a-1] = temp1 - self.lr * grad * temp1 / (temp1 + temp2)
            self.pi[s[1]:s_prime[1]+1, s[0]:s_prime[0]+1, a-2] = temp2 - self.lr * grad * temp2 / (temp1 + temp2)
        else:
            self.v[s[1]:s_prime[1]+1, s[0]:s_prime[0]+1] = self.v[s[1]:s_prime[1]+1, s[0]:s_prime[0]+1] + self.lr * grad
            self.pi[s[1]:s_prime[1]+1, s[0]:s_prime[0]+1, a] = pi + self.lr * grad

