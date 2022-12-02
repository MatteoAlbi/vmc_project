# vmc_project
Project for the "Vehicle Mechatronics: Control" course at Aalto University.
Our TurtleBot is required to complete a pre-fixed track without colliding with the walls, possibly travelling as fast as possible. 

The Gazebo environment is the following:
![image](https://user-images.githubusercontent.com/90208924/205381215-ccf6ac55-7105-488a-b116-c23437666e6c.png)

Our solution implements a Potential Field path planning, mostly using data coming from LiDaR sensor. Lateral control is performed with the Pure Pursuit algorithm.
The visualization of the potential field and of the trajectory is done using the Rviz widget.
![Immagine 2022-12-02 223201](https://user-images.githubusercontent.com/90208924/205380897-29b61d9c-46ac-473b-8418-349af1ef2766.png)

The package shall be cloned/downloaded in the catkin_ws/src folder, and after that the following commands are requested:
$ catkin_make
$ source ~/.bashrc

To launch the world:
$ roslaunch vmc_project project_world.launch

To run the node which starts the TurtleBot and the controller:
$ rosrun vmc_project racer.py
