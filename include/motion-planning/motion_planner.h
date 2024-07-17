#ifndef MOTION_PLANNER_H
#define MOTION_PLANNER_H

#include "motion-planning/map.h"
#include "motion-planning/robot.h"
#include "motion-planning/algorithms/base_algorithm.h"
#include "motion-planning/algorithms/cuda/base_algorithm_cuda.h"

class MotionPlanner {
public:
    MotionPlanner(Map &map, Robot &robot, BaseAlgorithm *algorithm);
    MotionPlanner(Map &map, Robot &robot, BaseAlgorithmCUDA *cudaAlgorithm);
    void planPath();
    void setAlgorithm(BaseAlgorithm *algorithm);
    void setAlgorithm(BaseAlgorithmCUDA *cudaAlgorithm);

private:
    Map &map;
    Robot &robot;
    BaseAlgorithm *algorithm;
    BaseAlgorithmCUDA *cudaAlgorithm;
    bool useCUDA;
};

#endif // MOTION_PLANNER_H
