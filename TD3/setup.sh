export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11411
export ROS_PORT_SIM=11411
export GAZEBO_RESOURCE_PATH=~/DRL-robot-navigation/catkin_ws/src/multi_robot_scenario/launch
source ~/.bashrc
cd ~/DRL-robot-navigation/catkin_ws
source devel/setup.bash
cd ~/DRL-robot-navigation/TD3
