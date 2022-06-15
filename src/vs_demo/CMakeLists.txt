cmake_minimum_required(VERSION 2.8.3)
project(vs_demo)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

set(BOOST_FLAG TRUE)
if(BOOST_FLAG)
    add_definitions( -DUSE_BOOST )
endif()

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  apriltags_ros
  geometry_msgs
  message_filters
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
  tm_msgs
)

## System dependencies are found with CMake's conventions
LIST(APPEND CMAKE_MODULE_PATH "/usr/share/cmake-2.8/Modules")
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
find_package(Boost REQUIRED COMPONENTS system thread chrono)

###################################
## catkin specific configuration ##
###################################
if(BOOST_FLAG)
catkin_package(
    INCLUDE_DIRS include
    #  LIBRARIES holo_demo
    LIBRARIES TM_REFLEXXESII_LIB RMLTYPEII_LIB   
    CATKIN_DEPENDS apriltags_ros geometry_msgs message_filters roscpp rospy sensor_msgs std_msgs tf tm_msgs
    DEPENDS boost
    #  DEPENDS system_lib
)
else()
catkin_package(
    INCLUDE_DIRS include
    #LIBRARIES TM_REFLEXXESII_LIB RMLTYPEII_LIB   
    CATKIN_DEPENDS apriltags_ros geometry_msgs message_filters roscpp rospy sensor_msgs std_msgs tf tm_msgs 
)
endif()

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)
set(CMAKE_CXX_FLAGS "-std=c++11")


add_executable(control_final src/control_final.cpp)
target_link_libraries(control_final
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
)