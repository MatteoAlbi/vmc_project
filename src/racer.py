#! /usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

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
        self.pot_stamp = 0
        
        self.pub_cloud = rospy.Publisher("sensor_msgs/PointCloud2", PointCloud2, queue_size=2)

        if self.need_plot:
            _, self.ax = plt.subplots(1,2, figsize=(12,6))
            self.ax[1] = sb.heatmap(self.pot_field, cbar=False)
            self.ax[1].invert_yaxis()        

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
                    dist_sqr = (x-p[0])**2 + (y-p[1])**2
                    self.pot_field[i,j] += self.pot_gain / dist_sqr
                self.pot_field[i,j] = min(self.pot_field[i,j], self.pot_cap)

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
        header.stamp = self.pot_stamp

        pc2 = point_cloud2.create_cloud(header, fields, xyzrgba)
        #print(pc2)

        #msg.is_bigendian = False
        #msg.point_step = 16
        #msg.row_step = msg.point_step * self.field_grid_y
        #msg.is_dense = True
        #msg.data = xyzrgba.tostring()

        self.pub_cloud.publish(pc2)

    def rviz_field(self):
        pot = np.asarray(self.pot_field).reshape(-1)

        # create x,y,z coordinate
        x = np.repeat(self.pot_x_points, self.field_grid_y).astype(np.float32)
        y = np.tile(self.pot_y_points, self.field_grid_x).astype(np.float32)
        z = (np.log(pot + 1) / 4).astype(np.float32)

        xyz = np.transpose(np.vstack((x,y,z)))

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]

        header = Header()
        header.frame_id = "pot_field"
        header.stamp = self.pot_stamp

        pc2 = point_cloud2.create_cloud(header, fields, xyz)
        self.pub_cloud.publish(pc2)
    
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

        #print("Computed trajectory:\n",self.trajectory,"\n")
        if self.need_plot:
            self.rviz_field()    


        return self.traj




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

        self.PotField.pot_stamp = rospy.Time.now()

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
        self.error = 1/radius
        self.cumulative_error = self.error * self.sample_time
        self.vel.angular.z =  self.Kp * self.error + \
                             self.Kd * (self.error - self.prev_error)/self.sample_time +\
                             self.Ki * (self.cumulative_error)
        self.prev_error = self.error

        brake = 1 + np.arctan2(yt[step], xt[step])      
        self.vel.linear.x /= brake                                                       

        self.pub.publish(self.vel)


    def __init__(self):
        self.sample_time = 0.1
        self.out = False
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=1/self.sample_time)
        self.vel = Twist()
        self.vel.linear.x = 0
        self.vel.angular.x = 0
        self.vel.angular.y = 0
        self.vel.angular.z = 0
        self.pub.publish(self.vel)
        self.rate = rospy.Rate(1/self.sample_time) 

        self.x = 0
        self.y = 0
        self.yaw = 0

        self.lidar_readings = []
        self.PotField = PotentialField(need_plot=True)


        self.Kp = 2
        self.Kd = 1
        self.Ki = 0
        self.error = 0
        self.prev_error = 0
        self.cumulative_error = 0

        self.look_ahead = 0.2
        self.k_brake = 0.1

        rospy.Subscriber('odom', Odometry, self.call_position)
        rospy.Subscriber('scan', LaserScan, self.call_Lidar)

        
        while not rospy.is_shutdown() and not self.out:
            
            self.PotField.lidar_readings = self.lidar_readings
            self.PotField.pot_stamp = rospy.get_rostime()
            self.PotField.create_field()
            traj = self.PotField.make_trajectory()
            #self.pure_pursuit(traj)

            #print("Commanded velocity: "+str(self.vel.angular.z))
            #print("Yaw: "+str(self.yaw))
            
            if len(self.lidar_readings) > 0:
                if self.lidar_readings[270] == float("inf") or self.lidar_readings[90] == float("inf"):
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