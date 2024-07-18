cmake_minimum_required(VERSION 3.0.2)
project(motion-planning)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# for ROS
#find_package(catkin REQUIRED COMPONENTS
#  roscpp
#  std_msgs
#)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# print out hthe OpenCV version and location
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

#catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES motion_planning
#  CATKIN_DEPENDS roscpp std_msgs
#  DEPENDS OpenCV
#)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

cuda_add_library(motion-planning
  src/map.cpp
  src/robot.cpp
  src/motion_planner.cpp
  src/algorithms/a_star.cpp
  src/algorithms/rrt.cpp
  src/algorithms/cuda/base_algorithm_cuda.cu
  src/algorithms/cuda/a_star_cuda.cu
)

# compile motion_planner as a library 
add_library(motion_planner src/motion_planner.cpp)
target_link_libraries(motion_planner motion-planning ${OpenCV_LIBS})

#add_executable(motion_planner src/motion_planner.cpp)
#target_link_libraries(motion_planner motion_planning ${OpenCV_LIBS})

# enable test
enable_testing()

# include gtest 
find_package(GTest REQUIRED)

# add test executable
add_executable(test_map tests/test_map.cpp)
target_link_libraries(test_map motion-planning ${OpenCV_LIBS} GTest::GTest GTest::Main)

add_executable(test_robot tests/test_robot.cpp)
target_link_libraries(test_robot motion-planning ${OpenCV_LIBS} GTest::GTest GTest::Main)

# add tests
add_test(NAME test_map COMMAND test_map) 
add_test(NAME test_robot COMMAND test_robot)
