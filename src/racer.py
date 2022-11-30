#! /usr/bin/env python3

from re import S
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

class PotentialField:

    def __init__(self, need_plot=False):

        self.lidar_readings = []
        self.x = 0
        self.y = 0
        self.lidar_threshold = 2
        self.field_x = 2
        self.field_y = 1
        self.field_offset_x = 0
        self.field_offset_y = 0.5
        self.field_grid_x = 20
        self.field_grid_y = 20
        self.pot_gain = 1
        self.traj_idx = []
        self.pot_cap = 15000
        self.need_plot = need_plot
        self.pot_field = np.zeros((self.field_grid_x,self.field_grid_y))
        self.attr_point = [self.field_x, 0]
        self.attr_gain = 1000


        if self.need_plot:
            _, self.ax = plt.subplots(1,2, figsize=(12,6))
            self.ax[1] = sb.heatmap(self.pot_field, cbar=False)
            self.ax[1].invert_yaxis()


    def plot_vision(self):
        x, y = zip(*self.cartesian)
        self.ax[0].clear()
        self.ax[0].scatter([ -i for i in list(y)],x)
        self.ax[0].scatter(0,0)
        self.ax[0].set_xlim([-self.field_offset_y, self.field_y-self.field_offset_y])
        self.ax[0].set_ylim([-self.lidar_threshold, self.lidar_threshold])
        self.ax[0].grid()
        y_tick = np.linspace(0-self.field_offset_x, self.field_x-self.field_offset_x, self.field_grid_x)
        x_tick = np.linspace(0-self.field_offset_y, self.field_y-self.field_offset_y, self.field_grid_y)
        y_tick = np.around(y_tick,decimals=2)[::-1]
        x_tick = np.around(x_tick,decimals=2)[::-1]
        #We flip on the first axis since seaborn would plot it tother way.
        #We flip on the second axis since the y direction is positive to the left, in the opposite way as the arraaaay would go.
        self.ax[1] = sb.heatmap(self.pot_field[::-1,::-1], xticklabels=x_tick, yticklabels=y_tick, cbar=False)#, ax=self.ax[1])
        plt.pause(0.0001)
        

    def create_field(self):
        self.cartesian = []
        self.pot_field = np.zeros((self.field_grid_x,self.field_grid_y))

        # Polar to Cartesian                                   
        for i, d in enumerate(self.lidar_readings):
            if d <= self.lidar_threshold:
                self.cartesian.append([np.cos(i*np.pi/180)*d, np.sin(i*np.pi/180)*d])

        # Potential Field
        self.pot_x_points = np.linspace(0-self.field_offset_x, self.field_x-self.field_offset_x, self.field_grid_x)
        self.pot_y_points = np.linspace(0-self.field_offset_y, self.field_y-self.field_offset_y, self.field_grid_y)
        for i,x in enumerate(self.pot_x_points):
            for j,y in enumerate(self.pot_y_points):
                for p in self.cartesian:
                    dist_sqr = (x-p[0])**2 + abs(y-p[1])#**2
                    self.pot_field[i,j] += self.pot_gain / (dist_sqr)
                dist_sqr = abs(x-self.attr_point[0]) + (y-self.attr_point[1])**2
                self.pot_field[i,j] -= self.attr_gain / np.sqrt(dist_sqr)
                self.pot_field[i,j] = min(self.pot_field[i,j], self.pot_cap)
                self.pot_field[i,j] = max(self.pot_field[i,j], -self.pot_cap)
    
    
    def make_trajectory(self, prev_traj):
        # trajectory
        finish = False
        #offset = [(1,0),(0,1),(1,1),(1,-1),(0,-1)]
        offset = [(1,0),(1,1),(1,-1)]
        self.traj_idx = []
        curr_point = [0,int(self.field_grid_y/2)]
        self.traj_idx.append(curr_point)
        
        point_count = 0
        while not finish and point_count < int(self.field_grid_x/1.5):

            #i_min = np.argmin([ self.pot_field[curr_point[0] + offset[0][0], curr_point[1] + offset[0][1]],\
            #                    self.pot_field[curr_point[0] + offset[1][0], curr_point[1] + offset[1][1]],\
            #                    self.pot_field[curr_point[0] + offset[2][0], curr_point[1] + offset[2][1]],\
            #                    self.pot_field[curr_point[0] + offset[3][0], curr_point[1] + offset[3][1]],\
            #                    self.pot_field[curr_point[0] + offset[4][0], curr_point[1] + offset[4][1]] ])
            
            i_min = np.argmin([ self.pot_field[curr_point[0] + offset[0][0], curr_point[1] + offset[0][1]],\
                                self.pot_field[curr_point[0] + offset[1][0], curr_point[1] + offset[1][1]],\
                                self.pot_field[curr_point[0] + offset[2][0], curr_point[1] + offset[2][1]] ])

            self.pot_field[curr_point[0],curr_point[1]] += self.pot_cap
            curr_point = [curr_point[0] + offset[i_min][0], curr_point[1] + offset[i_min][1]]
            self.traj_idx.append(curr_point)

            if curr_point[0] == self.field_grid_x-1 or curr_point[1] == 0 or curr_point[1] == self.field_grid_y-1:
                finish = True
            point_count+=1

        trajectory = np.array([[self.pot_x_points[x[0]], self.pot_y_points[x[1]]] for x in self.traj_idx])
        trajectory = np.around(trajectory, decimals=2)
        self.attr_point = trajectory[-1]
        

        #print("Computed trajectory:\n",trajectory,"\n")
        if self.need_plot:
            self.plot_vision()    

        return trajectory



class TurtleBot:

    def call_position(self, msg):
        euler_angles = euler_from_quaternion(  [msg.pose.pose.orientation.x,
                                                msg.pose.pose.orientation.y,
                                                msg.pose.pose.orientation.z,
                                                msg.pose.pose.orientation.w])
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = euler_angles[2]

    def call_Lidar(self, msg):
        self.lidar_readings = msg.ranges
        self.front_reading = msg.ranges[0]
        self.left_reading = msg.ranges[90]
        self.right_reading = msg.ranges[270]

    def avoid_walls(self):
        left_space = np.min([np.abs(self.left_reading - self.y),1])
        right_space = np.min([np.abs(self.right_reading - self.y),1])
        self.error =  np.max([np.min([left_space - right_space,1]),-1])
        self.cumulative_error += self.error * self.sample_time

        self.vel.angular.z =  self.Kp * self.error + \
                              self.Kd * (self.error - self.prev_error)/self.sample_time +\
                              self.Ki * (self.cumulative_error)

        self.prev_error = self.error
        #self.pub.publish(self.vel)

    def pure_pursuit(self,traj):
        xt = list(x[0] for x in traj)
        yt = list(x[1] for x in traj)

        d_arc = 0
        step = 0
        while d_arc < self.look_ahead and step <= len(xt):
            d_arc += np.sqrt((xt[step+1] - xt[step])**2 + (yt[step+1] - yt[step])**2)
            step += 1

        L_sq = (xt[step])**2 + (yt[step])**2
        radius = L_sq / (2 * yt[step])

        #print("Steer Input: ",1/radius)

        self.error = 1/radius
        self.cumulative_error = self.error * self.sample_time
        self.vel.angular.z =  self.Kp * self.error + \
                             self.Kd * (self.error - self.prev_error)/self.sample_time +\
                             self.Ki * (self.cumulative_error)
        #print("Error is: "+str(left_space - right_space))
        #print("Derivative Error is: "+str((self.error - self.prev_error)/self.sample_time))
        #print("Cumulative Error is: "+str(self.cumulative_error))                                     
        if len(self.lidar_readings) >0:
            large_cone = np.array([*self.lidar_readings[-35:], *self.lidar_readings[:35]])
            large_dist = np.min(large_cone)
            ahead_dist = np.min(large_cone[28:32])

            brake = 1 + np.abs(np.arctan2(yt[step], xt[step]))\
                    + self.k_brake_large* 1/(large_dist)\
                    + self.k_brake_ahead* 1/(ahead_dist)
            self.vel.linear.x = self.max_v / brake                                                      

        self.prev_error = self.error
        self.pub.publish(self.vel)

    def exit_control(self):
        self.out = True
        for i in range(45, 90, 1):
            if self.lidar_readings[i] != float("inf"):
                self.out = False
        
        for i in range(270, 315, 1):
            if self.lidar_readings[i] != float("inf"):
                self.out = False
            

        if self.out:
            self.vel.linear.x = 0
            self.vel.angular.z = 0
            self.pub.publish(self.vel)
            print(".............Can't find left/right wall, exiting...........")
        
    def __init__(self):
        self.sample_time = 0.05
        self.out = False

        self.x = 0
        self.y = 0
        self.yaw = 0

        self.lidar_readings = []
        self.front_reading = 0
        self.left_reading = 0
        self.right_reading = 0

        self.Kp = 0.58
        self.Kd = 0.03
        self.Ki = 0.0075
        self.error = 0
        self.prev_error = 0
        self.cumulative_error = 0
        self.max_v = 1.4

        self.look_ahead = 0.5
        self.traj = []
        self.k_brake_large = 0.26
        self.k_brake_ahead = 1.2

        #rospy.Subscriber('odom', Odometry, self.call_position)
        rospy.Subscriber('scan', LaserScan, self.call_Lidar)

        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=1/self.sample_time)
        self.vel = Twist()
        self.vel.linear.x = self.max_v 
        self.vel.angular.x = 0
        self.vel.angular.y = 0
        self.vel.angular.z = 0
        self.pub.publish(self.vel)
        self.rate = rospy.Rate(1/self.sample_time) 

        PotField = PotentialField(need_plot=False)

        while not rospy.is_shutdown() and not self.out:
            
            PotField.lidar_readings = self.lidar_readings
            PotField.create_field()
            self.traj = PotField.make_trajectory(self.traj)
            self.pure_pursuit(self.traj)
            if len(self.lidar_readings) > 0:
                self.exit_control()

            #print("Commanded velocity: "+str(self.vel.angular.z))
            #print("Yaw: "+str(self.yaw)
            
            self.rate.sleep()

if __name__ == '__main__':
    rospy.init_node('driver')
    try:
        Tbot = TurtleBot()          

    except rospy.ROSInterruptException: pass