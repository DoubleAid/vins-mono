#

## 如何在docker 内运行

1. 将代码挂在到容器中

```bash
docker run -idt -v ~/Code/ROS/vins-mono/:/app/catkin_ws/src osrf/ros2:devel
```

2. 查看ros的版本信息

```bash
# 察看环境变量
echo $ROS_DISTRO

# 查办版本信息
echo $ROS_VERSION

# 激活ROS环境
source /opt/ros/$ROS_DISTRO/setup.bash

# 运行ros
roscore
```

## 编译

1. 首先需要安装一些依赖库

```bash
apt update
apt install ros-noetic-tf ros-noetic-cv-bridge ros-noetic-message-filters ros-noetic-image-transport

# 安装 eigen 和 ceres
apt-get install libeigen3-dev
apt-get install libceres-dev

# 编译
catkin_make

# 运行
source devel/setup.bash

# 进入 build 目录

roslaunch vins_estimator euroc.launch 
roslaunch vins_estimator vins_rviz.launch
rosbag play YOUR_PATH_TO_DATASET/MH_01_easy.bag 
```
