#! /usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from std_msgs.msg import Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray 
from geometry_msgs.msg import Point 

from tf.transformations import euler_from_quaternion
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

BOOL_PLOT = True

class PotentialField:

    def __init__(self, need_plot=False):

        self.lidar_readings = []
        # threshold for the lidar: further points are not considered
        self.lidar_threshold = 2 
        # cap for the potential: higher values are capped to this value
        self.pot_cap = 15000

        # field dimensions
        self.field_x = 2
        self.field_y = 1
        # field position offset wrt robot (offset = 0 -> robot in bottom left corner)
        self.field_offset_x = 0
        self.field_offset_y = 0.5
        # number of points per axis
        self.field_grid_x = 20
        self.field_grid_y = 20
        # create grid for the potential
        self.pot_x_points = np.linspace(0-self.field_offset_x, self.field_x-self.field_offset_x, self.field_grid_x)
        self.pot_y_points = np.linspace(0-self.field_offset_y, self.field_y-self.field_offset_y, self.field_grid_y)

        # gain for the potential: pot = gain/dist(x,y)
        self.pot_gain = 1 # gain for repulsive points
        self.attr_gain = 1000 # gain for attraction point
        
        # store computed trajectory
        self.traj_idx = []
        # bool variable to display potential 
        self.need_plot = need_plot

        # initialization
        self.pot_field = np.zeros((self.field_grid_x,self.field_grid_y))
        self.attr_point = [self.field_x, 0]
        self.time_stamp = 0
        
        self.pub_cloud = rospy.Publisher("rviz_racer/PointCloud2", PointCloud2, queue_size = 2)
        self.pub_marker = rospy.Publisher("rviz_racer/Marker", MarkerArray, queue_size = 2)
        self.mark_arr = MarkerArray()

        # header of the point cloud
        self.header_pcl = Header()
        self.header_pcl.frame_id = "map"

        # marker of the attraction point
        self.marker_attp = Marker()
        self.marker_attp.header.frame_id = "map"
        self.marker_attp.type = 2 #sphere
        self.marker_attp.id = 0
        # set scale
        self.marker_attp.scale.x = 0.1
        self.marker_attp.scale.y = 0.1
        self.marker_attp.scale.z = 0.01
        # set color
        self.marker_attp.color.r = 0.0
        self.marker_attp.color.g = 1.0
        self.marker_attp.color.b = 0.0
        self.marker_attp.color.a = 1.0
        # set pose
        self.marker_attp.pose.position.z = 0
        self.marker_attp.pose.orientation.x = 0.0
        self.marker_attp.pose.orientation.y = 0.0
        self.marker_attp.pose.orientation.z = 0.0
        self.marker_attp.pose.orientation.w = 1.0

        # marker for the trajectory
        self.marker_traj = Marker()
        self.marker_traj.header.frame_id = "map"
        self.marker_traj.type = Marker.LINE_STRIP
        self.marker_traj.action = Marker.ADD
        self.marker_traj.id = 1
        # set scale
        self.marker_traj.scale.x = 0.02
        # set color
        self.marker_traj.color.r = 1.0
        self.marker_traj.color.g = 0.0
        self.marker_traj.color.b = 0.0
        self.marker_traj.color.a = 0.5
        # set pose
        self.marker_traj.pose.position.x = 0
        self.marker_traj.pose.position.y = 0
        self.marker_traj.pose.position.z = 0
        self.marker_traj.pose.orientation.x = 0.0
        self.marker_traj.pose.orientation.y = 0.0
        self.marker_traj.pose.orientation.z = 0.0
        self.marker_traj.pose.orientation.w = 1.0

        # if self.need_plot:
        #     # plot window
        #     _, self.ax = plt.subplots(1,2, figsize=(12,6))
        #     self.ax[1] = sb.heatmap(self.pot_field, cbar=False)
        #     self.ax[1].invert_yaxis()

    def create_field(self):
        self.cartesian = []
        self.pot_field = np.zeros((self.field_grid_x,self.field_grid_y))

        # Polar to Cartesian                                   
        for i, d in enumerate(self.lidar_readings):
            if d <= self.lidar_threshold:
                self.cartesian.append([np.cos(i*np.pi/180)*d, np.sin(i*np.pi/180)*d])

        # Potential Field
        for i,x in enumerate(self.pot_x_points): # x coordinate on potential grid
            for j,y in enumerate(self.pot_y_points): # y coordinate of potential grid
                for p in self.cartesian: # lidar points
                    # distance function
                    dist_sqr = (x-p[0])**2 + abs(y-p[1])#**2
                    # potential contribute of the point
                    self.pot_field[i,j] += self.pot_gain / (dist_sqr)
                
                # attractive point potential
                dist_sqr = abs(x-self.attr_point[0]) + (y-self.attr_point[1])**2
                self.pot_field[i,j] -= self.attr_gain / np.sqrt(dist_sqr)
                
                # apply potential cap
                self.pot_field[i,j] = min(self.pot_field[i,j], self.pot_cap)
                self.pot_field[i,j] = max(self.pot_field[i,j], -self.pot_cap)

    def make_trajectory(self):
        finish = False
        # offset in grid coordinates
        offset = [(1,0),(1,1),(1,-1)]
        self.traj_idx = []
        # current position of the robot in the potential grid
        curr_point = [int( self.field_grid_x * self.field_offset_x / self.field_x), int( self.field_grid_y * self.field_offset_y / self.field_y)]
        self.traj_idx.append(curr_point)
        
        point_count = 0 # limit number of point of trajectory as function of number of point in x direction
        while not finish and point_count < int(self.field_grid_x/1.5):
            # check potential in front points, choose point with lowest potential            
            i_min = np.argmin([ self.pot_field[curr_point[0] + offset[0][0], curr_point[1] + offset[0][1]],\
                                self.pot_field[curr_point[0] + offset[1][0], curr_point[1] + offset[1][1]],\
                                self.pot_field[curr_point[0] + offset[2][0], curr_point[1] + offset[2][1]] ])
            # increment curr point before switching to visualize trajectory
            #if self.need_plot: self.pot_field[curr_point[0],curr_point[1]] += self.pot_cap
            # switch point and add to traj
            curr_point = [curr_point[0] + offset[i_min][0], curr_point[1] + offset[i_min][1]]
            self.traj_idx.append(curr_point)

            # termination condition: extremes of the grid
            if curr_point[0] == self.field_grid_x-1 or curr_point[1] == 0 or curr_point[1] == self.field_grid_y-1:
                finish = True
            point_count += 1

        # switch from grid coordinates to local coordinates
        trajectory = np.array([[self.pot_x_points[x[0]], self.pot_y_points[x[1]]] for x in self.traj_idx])
        # set attraction point as last point of the trajectory
        self.attr_point = trajectory[-1]
        
        #trajectory = np.around(trajectory, decimals=2)
        #print("Computed trajectory:\n",self.trajectory,"\n")

        if self.need_plot:
            self.rviz_plot()    

        return trajectory

# NOT USED: TOO HEAVY COMPUTATIONS
    def plot_heatmap(self):
        # unpack x and y coordinates for each point
        x, y = zip(*self.cartesian)
        self.ax[0].clear()
        # cartesian coordinates of the lidar readings
        self.ax[0].scatter([ -i for i in list(y)],x)
        self.ax[0].scatter(0,0)
        self.ax[0].set_xlim([-self.field_offset_y, self.field_y-self.field_offset_y])
        self.ax[0].set_ylim([-self.lidar_threshold, self.lidar_threshold])
        self.ax[0].grid()
        # heatmap of the potential
        y_tick = np.linspace(0-self.field_offset_x, self.field_x-self.field_offset_x, self.field_grid_x)
        x_tick = np.linspace(0-self.field_offset_y, self.field_y-self.field_offset_y, self.field_grid_y)
        
        x_tick = np.around(x_tick,decimals=2)[::-1] # We flip on the first axis since seaborn would plot it the other way
        y_tick = np.around(y_tick,decimals=2)[::-1] # We flip on the second axis since the y direction is positive to the left, in the opposite way as the array would go
        
        #plot
        self.ax[1] = sb.heatmap(self.pot_field[::-1,::-1], xticklabels=x_tick, yticklabels=y_tick, cbar=False)#, ax=self.ax[1])
        plt.pause(0.0001)
        
# NOT USED, NEED FURTHER DEVELOP TO PROPERLY SET COLOR
    def rviz_field_color(self): 
        pot = np.asarray(self.pot_field).reshape(-1)

        # create x,y,z coordinate
        x = np.repeat(self.pot_x_points, self.field_grid_y).astype(np.float32)
        y = np.tile(self.pot_y_points, self.field_grid_x).astype(np.float32)
        z = (np.log(pot + 1) / 4).astype(np.float32)
        
        # create color field
        #r = (255 - pot/self.pot_cap*255).astype(np.ubyte)
        #b = (pot/self.pot_cap*255).astype(np.ubyte)
        r = np.ones(pot.shape, np.ubyte) * 0
        g = np.ones(pot.shape, np.ubyte) * 0
        b = np.ones(pot.shape, np.ubyte) * 255
        a = np.ones(pot.shape, np.ubyte) * 0
        # convert to UINT32
        rgba = np.hstack((r,g,b,a))
        rgba = rgba.view("uint32")

        # Structured array
        xyzrgba = np.zeros( (x.shape), \
            dtype={
                "names": ("x", "y", "z", "rgba"),
                "formats": ("f4", "f4", "f4", "u4")} ) 

        xyzrgba["x"] = x
        xyzrgba["y"] = y
        xyzrgba["z"] = z
        xyzrgba["rgba"] = rgba

        #xyzrgba = np.transpose(np.vstack((x,y,z,r,g,b,a)))
        #xyz = np.transpose(np.vstack((x,y,z)))

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1)
        ]

        header = Header()
        header.frame_id = "pot_field"
        header.stamp = self.time_stamp

        pc2 = point_cloud2.create_cloud(header, fields, xyzrgba)
        #print(pc2)

        #msg.is_bigendian = False
        #msg.point_step = 16
        #msg.row_step = msg.point_step * self.field_grid_y
        #msg.is_dense = True
        #msg.data = xyzrgba.tostring()

        self.pub_cloud.publish(pc2)

    def rviz_plot(self):
        ## Cloudpoint of the potential field
        pot = np.asarray(self.pot_field).reshape(-1)
        # create x,y,z coordinate
        x = np.repeat(self.pot_x_points, self.field_grid_y).astype(np.float32)
        y = np.tile(self.pot_y_points, self.field_grid_x).astype(np.float32)
        #z = (pot + self.pot_cap)/30000*9999 + 1  # range 1 / 1k
        #z = np.log10(z) - 2 # range -2 / 2
        z = pot/10000
        # aggregate data
        xyz = np.transpose(np.vstack((x,y,z)))
        # header info
        self.header_pcl.stamp = self.time_stamp
        # create message and publish cloudpoint
        pc2 = point_cloud2.create_cloud_xyz32(self.header_pcl, xyz)
        self.pub_cloud.publish(pc2)

        ## Marker of the attraction point
        self.marker_attp.header.stamp = self.time_stamp
        # set position
        self.marker_attp.pose.position.x = self.attr_point[0]
        self.marker_attp.pose.position.y = self.attr_point[1]
        # append in marker array
        self.mark_arr.markers.append(self.marker_attp) 

        ## Display trajectory
        traj = np.array([[self.pot_x_points[x[0]], self.pot_y_points[x[1]]] for x in self.traj_idx])
        # append trajectory points to marker
        self.marker_traj.points=[] 
        for i in range(traj.shape[0]): 
            p = Point()
            p.x = traj[i,0]
            p.y = traj[i,1]
            p.z = 0
            self.marker_traj.points.append(p) 
        # append in marker array
        self.mark_arr.markers.append(self.marker_traj)

        # publish
        self.pub_marker.publish(self.mark_arr)
        # empty marker array
        self.mark_arr.markers = []

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

    def pure_pursuit(self,traj):
        # split trajectory coordinates (local frame)
        xt = list(x[0] for x in traj)
        yt = list(x[1] for x in traj)

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
        self.Kp = 0.55
        self.Kd = 0.04
        self.Ki = 0.01
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
        self.k_brake_ahead = 0.8 # may be rised

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
            self.PotField.create_field()
            self.traj = self.PotField.make_trajectory()
            # pure pursuit trajectory follower
            self.pure_pursuit(self.traj)
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