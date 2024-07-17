# Motion Planning Library
This is a motion planning library implemented in c++ and cuda, with integration of ROS.

## library structure

```lua
motion_planning/
|-- include/
|   |-- motion_planning/
|       |-- map.h
|       |-- robot.h
|       |-- motion_planner.h
|       |-- algorithms/
|           |-- base_algorithm.h
|           |-- a_star.h
|           |-- rrt.h
|           |-- cuda/
|               |-- base_algorithm_cuda.h
|               |-- a_star_cuda.h
|-- src/
|   |-- map.cpp
|   |-- robot.cpp
|   |-- motion_planner.cpp
|   |-- algorithms/
|       |-- base_algorithm.cpp
|       |-- a_star.cpp
|       |-- rrt.cpp
|       |-- cuda/
|           |-- base_algorithm_cuda.cu
|           |-- a_star_cuda.cu
|-- tests/
|   |-- map_test.cpp
|   |-- robot_test.cpp
|   |-- motion_planner_test.cpp
|-- CMakeLists.txt
|-- package.xml
|-- README.md
```
