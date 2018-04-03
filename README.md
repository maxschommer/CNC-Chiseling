# CNC-Chiseling

Clone this repo inside a catkin workspace. 

# Install fstl:

See documentation available at https://github.com/mkeeter/fstl.
If packages are not sourced, make sure to source them. For example, if your catkin workspace is in your Documents folder, run """source ~/Documents/catkin_ws_ind/devel/setup.sh""".

# Install MoveIt from source:
"""
roscd && cd ../
git clone git@github.com:ros-planning/moveit.git
sudo apt install ros-kinetic-moveit-resources
catkin_make
"""


# Run Rviz of Arm with motion planning:
 
"""
cd ./StaubliURDF/move_it_arm_2/launch/
roslaunch demo.launch use_gui:=true
"""

# Communicating with Staubli Arm
Install minicom https://help.ubuntu.com/community/Minicom
Setting it up http://developer.ridgerun.com/wiki/index.php/Setting_up_Minicom_Client_-_Ubuntu

1. Turn on Arm
2. Connect the AWCII RS232 TERM usb first (as /dev/ttyUSB0). Then connect the RS232 port from the main processer (as /dev/ttyUSB1)
3. 
```
sudo minicom -D /dev/ttyUSB0
``` 
4. Open arduino serial monitor (make sure the port is USB0) then close it (unsure of why this is needed)
5.
```
D (load from disk d)
Y (yes, on scribe marks)
abort 0 (abort background process)
``` 

6. exit minicom using ctrl-A Q
7.
```
python sendvplus.py 
```
8. 
disable dry.run
enable power

# Running Slicer and CAM

## Installs:

Required packages can be installed via the following commands:

"""
sudo apt install python-tk
sudo pip install numpy-stl bintrees sortedcontainers matplotlib shapely descartes
"""