#! /usr/bin/env python3 
 
from cmath import pi 
import rospy 
import numpy as np 
from geometry_msgs.msg import Twist 
from sensor_msgs.msg import LaserScan 
from nav_msgs.msg import Odometry 
from tf.transformations import euler_from_quaternion 
from visualization_msgs.msg import Marker
import numpy as np 
import seaborn as sb 
import matplotlib.pyplot as plt 
 
 
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
        print("Error is: "+str(left_space - right_space)) 
        self.cumulative_error += self.error * self.sample_time 
 
        self.vel.angular.z =  self.Kp * self.error + \ 
                              self.Kd * (self.error - self.prev_error)/self.sample_time +\ 
                              self.Ki * (self.cumulative_error) 
        self.prev_error = self.error 
 
    def trajectory(self): 
        # polar to cartesian 
        cartesian = [] 
        for i, d in enumerate(self.lidar_readings): 
            if d <= self.lidar_threshold: 
                cartesian.append([np.cos(i*pi/180)*d, np.sin(i*pi/180)*d]) 
 
        #print("Lidar points cartesian points:\n",cartesian,"\n")         
 
        # potential field 
        pot_field = np.zeros((int(self.field_x/0.05),int(self.field_y/0.05))) 
         
        for i,x in enumerate(range(int((0-self.field_offset_x)*20), int((self.field_x-self.field_offset_x)*20), 1)): 
            for j,y in enumerate(range(int((0-self.field_offset_y)*20), int((self.field_y-self.field_offset_y)*20), 1)): 
                for p in cartesian: 
                    dist_sqr = (x/20-p[0])**2 + (y/20-p[1])**2 
                    pot_field[i,j] += self.pot_gain / dist_sqr 
         
        #print("Potential field:\n",pot_field,"\n") 
        #plt.figure() 
        #sb.heatmap(pot_field) 
        #plt.show() 

        # trajectory 
        # finish = False 
        # offset = [(0,1),(1,1),(1,0),(1,-1),(0,-1)] 
 
        # self.traj = [] 
        # curr_point = [0,10] 
        # self.traj.append(curr_point) 
         
        # while not finish: 
        #     i_min = np.argmin([ pot_field[curr_point[0] + offset[0][0], curr_point[1] + offset[0][1]],\ 
        #                         pot_field[curr_point[0] + offset[0][0], curr_point[1] + offset[0][1]],\ 
        #                         pot_field[curr_point[0] + offset[0][0], curr_point[1] + offset[0][1]],\ 
        #                         pot_field[curr_point[0] + offset[0][0], curr_point[1] + offset[0][1]],\ 
        #                         pot_field[curr_point[0] + offset[0][0], curr_point[1] + offset[0][1]] ]) 
             
        #     curr_point = [curr_point[0] + offset[i_min][0], curr_point[1] + offset[i_min][1]] 
        #     self.traj.append(curr_point) 
 
        #     if curr_point[0] == 19 or curr_point[1] == 0 or curr_point[1] == 19: 
        #         finish = True 
 
        print("Robot Position: ", self.x, " ", self.y) 
        print("Computed trajectory:\n",self.traj,"\n") 
 
 
    def __init__(self): 
        self.sample_time = 1 
        self.out = False 
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=1/self.sample_time) 
        self.marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size = 2)
        self.vel = Twist() 
        self.vel.linear.x = 0.1 
        self.vel.angular.x = 0 
        self.vel.angular.y = 0 
        self.vel.angular.z = 0 
        self.pub.publish(self.vel) 
        self.rate = rospy.Rate(1/self.sample_time)  
 
        self.x = 0 
        self.y = 0 
        self.yaw = 0 
 
        self.lidar_readings = [] 
        self.lidar_threshold = 1 
        self.field_x = 1 
        self.field_y = 1 
        self.field_offset_x = 0 
        self.field_offset_y = 0.5 
        self.pot_gain = 1 
        self.traj = [] 
        self.marker = Marker()

        self.marker.header.frame_id = "/pot_field"
        self.marker.type = 1
        self.marker.id = 0
        # Set the scale of the marker
        self.marker.scale.x = 0.1
        self.marker.scale.y = 0.1
        self.marker.scale.z = 0.1
        self.marker.pose.orientation.x = 0.0
        self.marker.pose.orientation.y = 0.0
        self.marker.pose.orientation.z = 0.0
        self.marker.pose.orientation.w = 1.0
 
        self.front_reading = 0 
        self.left_reading = 0 
        self.right_reading = 0 
 
        self.Kp= 3 
        self.Kd = 1 
        self.Ki = 0.05 
        self.error = 0 
        self.prev_error = 0 
        self.cumulative_error = 0 
 
        rospy.Subscriber('odom', Odometry, self.call_position) 
        rospy.Subscriber('scan', LaserScan, self.call_Lidar) 
 
        while not rospy.is_shutdown() and not self.out: 
             
            self.avoid_walls() 
            self.trajectory() 
 
            print("Commanded velocity: "+str(self.vel.angular.z)) 
            #print("Yaw: "+str(self.yaw)) 
            self.pub.publish(self.vel) 
 
            if self.left_reading == float("inf") or self.right_reading == float("inf"): 
                self.out = True 
                self.vel.linear.x = 0 
                self.vel.angular.z = 0 
 
                self.pub.publish(self.vel) 
                print(".............Can't find left/right wall, exiting...........") 
 
            self.rate.sleep() 
 
 
 
if __name__ == '__main__': 
    rospy.init_node('driver') 
    try: 
        Tbot = TurtleBot()           
 
    except rospy.ROSInterruptException: pass