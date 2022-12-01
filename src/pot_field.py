#! /usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
import numpy as np



class PotentialField:

    def __init__(self, need_plot=False):

        self.lidar_readings = []
        self.x = 0
        self.y = 0
        self.lidar_threshold = 2
        self.field_x = 1
        self.field_y = 1
        self.field_offset_x = 0
        self.field_offset_y = 0.5
        self.field_grid_x = 20
        self.field_grid_y = 20
        self.pot_gain = 1
        self.traj = []
        self.pot_cap = 15000
        self.need_plot = need_plot
        self.pot_field = np.zeros((self.field_grid_x,self.field_grid_y))

        self.pot_x_points = np.linspace(0-self.field_offset_x, self.field_x-self.field_offset_x, self.field_grid_x)
        self.pot_y_points = np.linspace(0-self.field_offset_y, self.field_y-self.field_offset_y, self.field_grid_y)
        

        #if self.need_plot:
            #_, self.ax = plt.subplots(1,2, figsize=(12,6))
            #self.ax[1] = sb.heatmap(self.pot_field, cbar=False)
            #self.ax[1].invert_yaxis()
        

    def create_field(self):
        self.cartesian = []
        self.pot_field = np.zeros((self.field_grid_x,self.field_grid_y))

        # Polar to Cartesian                                   
        for i, d in enumerate(self.lidar_readings):
            if d <= self.lidar_threshold:
                self.cartesian.append([np.cos(i*np.pi/180)*d, np.sin(i*np.pi/180)*d])

        # Potential Field
        
        for i,x in enumerate(self.pot_x_points):
            for j,y in enumerate(self.pot_y_points):
                for p in self.cartesian:
                    dist_sqr = (x-p[0])**2 + (y-p[1])**2
                    self.pot_field[i,j] += self.pot_gain / dist_sqr
                self.pot_field[i,j] = min(self.pot_field[i,j], self.pot_cap)
        
    def plot_vision(self):
        x, y = zip(*self.cartesian)
        y_tick = np.linspace(0-self.field_offset_x, self.field_x-self.field_offset_x, self.field_grid_x)
        x_tick = np.linspace(0-self.field_offset_y, self.field_y-self.field_offset_y, self.field_grid_y)
        y_tick = np.around(y_tick,decimals=2)[::-1]
        x_tick = np.around(x_tick,decimals=2)[::-1]
        #We flip on the first axis since seaborn would plot it tother way.
        #We flip on the second axis since the y direction is positive to the left, in the opposite way as the arraaaay would go.
        #self.ax[1] = sb.heatmap(self.pot_field[::-1,::-1], xticklabels=x_tick, yticklabels=y_tick, cbar=False)#, ax=self.ax[1])
        #self.ax[1] = sb.lineplot(zip(*self.trajectory))
        #self.ax[1] = sb.heatmap(pot_field, cbar=False)#, ax=self.ax[1])
        #plt.pause(0.001)

    
    def make_trajectory(self):
        # trajectory
        finish = False
        offset = [(0,1),(1,1),(1,0),(1,-1),(0,-1)]
        self.traj = []
        curr_point = [0,10]
        self.traj.append(curr_point)
        
        point_count = 0
        while not finish and point_count <20:

            i_min = np.argmin([ self.pot_field[curr_point[0] + offset[0][0], curr_point[1] + offset[0][1]],\
                                self.pot_field[curr_point[0] + offset[1][0], curr_point[1] + offset[1][1]],\
                                self.pot_field[curr_point[0] + offset[2][0], curr_point[1] + offset[2][1]],\
                                self.pot_field[curr_point[0] + offset[3][0], curr_point[1] + offset[3][1]],\
                                self.pot_field[curr_point[0] + offset[4][0], curr_point[1] + offset[4][1]] ])
            curr_point = [curr_point[0] + offset[i_min][0], curr_point[1] + offset[i_min][1]]
            self.traj.append(curr_point)

            if curr_point[0] == 19 or curr_point[1] == 0 or curr_point[1] == 19:
                finish = True
            point_count+=1

        trajectory = np.array([[self.pot_x_points[x[0]], self.pot_y_points[x[1]] ] for x in self.traj])
        self.trajectory = np.around(trajectory, decimals=2)

        print("Computed trajectory:\n",self.trajectory,"\n")
        if self.need_plot:
            self.plot_vision()    


        return self.traj