#! /usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
#from nav_msgs.msg import Odometry

from tf.transformations import euler_from_quaternion

from pot_field import PotentialField

BOOL_PLOT = True

class TurtleBot:

    def call_position(self, msg):
        # get x,y,yaw of robot from subscriber
        euler_angles = euler_from_quaternion(  [msg.pose.pose.orientation.x,
                                                msg.pose.pose.orientation.y,
                                                msg.pose.pose.orientation.z,
                                                msg.pose.pose.orientation.w])
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = euler_angles[2]

    def call_Lidar(self, msg):
        # get lidar readings from subscriber
        self.lidar_readings = msg.ranges

        if self.PotField.need_plot: self.PotField.time_stamp = rospy.Time.now()

    def pure_pursuit(self):
        # split trajectory coordinates (local frame)
        xt = list(x[0] for x in self.traj)
        yt = list(x[1] for x in self.traj)

        d_arc = 0
        step = 0
        # move about look ahead distance
        while d_arc < self.look_ahead and step <= len(xt):
            d_arc += np.sqrt((xt[step+1] - xt[step])**2 + (yt[step+1] - yt[step])**2)
            step += 1

        # obtain radius: all coordinates are already in local frame
        L_sq = (xt[step])**2 + (yt[step])**2
        radius = L_sq / (2 * yt[step])

        # yaw = 0 in local frame
        self.error = 1/radius
        # apply PID control to angular vel
        self.cumulative_error = self.error * self.sample_time
        self.vel.angular.z =  self.Kp * self.error + \
                             self.Kd * (self.error - self.prev_error)/self.sample_time +\
                             self.Ki * (self.cumulative_error)
        self.prev_error = self.error
        
        # limiter to speed
        if len(self.lidar_readings) >0:
            large_cone = np.array([*self.lidar_readings[-30:], *self.lidar_readings[:30]])
            large_dist = np.min(large_cone) # closest point distance on the sides
            ahead_dist = np.min(large_cone[25:35]) # closest point distance in the front

            # compute brake factor
            brake = 1 + np.abs(np.arctan2(yt[step], xt[step])) # deviation from desired heading direction
            brake += self.k_brake_large* 1/(large_dist) # sides distance factor
            brake += self.k_brake_ahead* 1/(ahead_dist) # front distance factor
            self.vel.linear.x = self.max_v / brake      # limiting speed                                               

        
        self.pub.publish(self.vel)

    def exit_control(self):
        self.out = True
        # control set of frontal points to asses if the track is complete

        for i in range(45, 90, 1):
            if self.lidar_readings[i] != float("inf"):
                self.out = False
        
        for i in range(270, 315, 1):
            if self.lidar_readings[i] != float("inf"):
                self.out = False
            
        # stop robot
        if self.out:
            self.vel.linear.x = 0
            self.vel.angular.z = 0
            self.pub.publish(self.vel)
            print(".............Can't find left/right wall, exiting...........")
        
    def __init__(self):
        
        self.sample_time = 0.1 # may be lowered
        self.rate = rospy.Rate(1/self.sample_time) 

        # robot pos+attitude and lidar
        self.x = 0
        self.y = 0
        self.yaw = 0

        # initialize potential field object to compute trajectory
        self.PotField = PotentialField(need_plot=BOOL_PLOT)
        self.lidar_readings = []

        # PID parameters
        # sub 1min
        self.Kp = 0.5
        self.Kd = 0.045
        self.Ki = 0.005
        # sub 50s: unstable
        # self.Kp = 0.58
        # self.Kd = 0.03
        # self.Ki = 0.0075
        self.error = 0
        self.prev_error = 0
        self.cumulative_error = 0

        # pure pursuit parameters
        self.max_v = 1.3
        self.look_ahead = 0.5
        self.traj = []
        self.k_brake_large = 0.4 # may be lowered 
        self.k_brake_ahead = 1.1 # may be rised

        # subscriber to retrieve x, y, yaw and lidar readings
        #rospy.Subscriber('odom', Odometry, self.call_position) # not used
        rospy.Subscriber('scan', LaserScan, self.call_Lidar)

        # control velocity message
        self.vel = Twist()
        self.vel.linear.x = 0
        self.vel.angular.x = 0
        self.vel.angular.y = 0
        self.vel.angular.z = 0

        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size = 1/self.sample_time)
        self.pub.publish(self.vel)

        self.out = False # stop boolean
        
        # running loop
        while not rospy.is_shutdown() and not self.out: 
            # compute trajectory using potential field
            self.PotField.lidar_readings = self.lidar_readings
            if self.PotField.need_plot: self.PotField.time_stamp = rospy.get_rostime()
            self.traj = self.PotField.make_trajectory()
            
            # pure pursuit trajectory follower
            self.pure_pursuit()
            # check stop condition
            if len(self.lidar_readings) > 0:
                self.exit_control()
            
            self.rate.sleep()

if __name__ == '__main__':
    # run node
    rospy.init_node('driver')
    try:
        Tbot = TurtleBot()          

    except rospy.ROSInterruptException: pass