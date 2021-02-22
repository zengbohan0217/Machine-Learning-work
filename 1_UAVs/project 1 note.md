# Project 1 note

### Target

​		The goal of this task is to use different methods to realize the obstacle avoidance and path-finding task of UAV, The algorithm used in this project is Dijkstra algorithm.

### File Declaration

+ **UAV.py**: The class UAV is defined in this file. The function of obtaining the full graph path risk coefficient and using Dijkstra algorithm to obtain the next target point is encapsulated here。
+ **planning.py**: The most critical function is to obtain the list of UAV running paths according to the starting point and destination.
+ **enemy_record.py**: Record enemy coordinates and other parameters.
+ **get_dis.py**: This file contains simple functions for calculating the distance between points, points and lines.
+ **draw_background.py**: The main purpose of this file is to map enemy gradients and the route of the UAV.
+ **main.py**: This file is used to run the entire task.