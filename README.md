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

# Running Slicer and CAM

## Installs:

Required packages can be installed via the following commands:

"""
sudo pip install numpy-stl bintrees sortedcontainers
"""
