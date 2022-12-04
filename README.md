# vmc_project
Project for the "Vehicle Mechatronics: Control" course at Aalto University.
Our TurtleBot is required to complete a pre-fixed track without colliding with the walls, possibly travelling as fast as possible. 

The Gazebo environment is the following:
![image](https://user-images.githubusercontent.com/90208924/205381215-ccf6ac55-7105-488a-b116-c23437666e6c.png)

## Configuration and running
The package shall be cloned/downloaded in the catkin_ws/src folder, and after that run the following commands from the *~/catkin* folder:
```
$ catkin_make
$ source ~/.bashrc
```
To launch the world:
```
$ roslaunch vmc_project project_world.launch
```
To run the node which starts the TurtleBot and the controller:
```
$ rosrun vmc_project racer.py
```
### Visualization
The visualization of the potential field and of the trajectory is done using the Rviz widget.

![Immagine 2022-12-02 223201](https://user-images.githubusercontent.com/90208924/205380897-29b61d9c-46ac-473b-8418-349af1ef2766.png)

To run the visualization run rviz and the command 
```
rosrun tf static_transform_publisher 0 0 0 0 0 0 1 map base_scan 10
``` 
(can be copied-pasted from [this file](rviz_cfg/set_rviz_frame_commands)).
Then from rviz open the configuration file [racer.rviz](rviz_cfg/racer.rviz).

To avoid the visualization and unnecessary computation, set to false the variable *BOOL_PLOT* in file [racer.py](src/racer.py). 


## Path planning
Our solution implements a Potential Field path planning, mostly using data coming from LiDAR sensor. The potential field comprehend two different contributions:
- Repulsive from the walls, represented by the points detected by the LiDAR. The function used for each point is: 
```python
# distance function
dist_sqr = (x-p[0])**2 + abs(y-p[1])
# potential contribute of the point
pot_field[i,j] += pot_gain / (dist_sqr)
```
- Attractive from a point positioned at the end of the trajectory computed at previous time step. The function used for the attractive point is:
```python
# distance function
dist_sqr = abs(x-att_p[0]) + (y-att_p[1])**2
# potential contribute of the point
pot_field[i,j] -= attr_gain / np.sqrt(dist_sqr)
```
With *x*,*y* point in which the potential field is computed, *pot_gain* magnitude constant of the walls repulsive field, *attr_gain* magnitude constant of the attractive field, *p* walls points and *att_p* position of the attraction point.

The resulting potential field is the following:
![Potential field](/media/geogebra-export.png)

Using this function we are able to incentivize a forward trajectory and overcome problems related to local minima of the potential (e.g. at the crossroad).

Tuned parameters are the attractive and repulsive gains, the functions used, and the position of the attraction point with respect to the previous trajectory.

## Lateral control
Lateral control is performed with the Pure Pursuit algorithm. The algorithm results simplified given that the trajectory is computed in the robot local reference frame. A PID controller is then implemented to adjust the yaw rate.

Tuned parameters are the PID gains and the look-ahead distance.

## Longitudinal control
The lognitudinal control is impemented setting a maximum speed the robot can achieve, then applying a braking factor compsed by three different components:
- Deviation from desired yaw angle, set by the lateral control.
- Closest point distance on lateral sides.
- Closest point distance in front of the robot.
  
Each contribution has its own gain. Tuned parameters are the braking gains and the max velocity.
