#include "motion-planning/motion_planner.h"

MotionPlanner::MotionPlanner(Map &map, Robot &robot, BaseAlgorithm *algorithm)
    : map(map), robot(robot), algorithm(algorithm), cudaAlgorithm(nullptr), useCUDA(false) {}

MotionPlanner::MotionPlanner(Map &map, Robot &robot, BaseAlgorithmCUDA *cudaAlgorithm)
    : map(map), robot(robot), algorithm(nullptr), cudaAlgorithm(cudaAlgorithm), useCUDA(true) {}

void MotionPlanner::planPath() {
    if (useCUDA) {
        if (cudaAlgorithm) {
            cudaAlgorithm->findPath(map, robot);
        }
    } else {
        if (algorithm) {
            algorithm->findPath(map, robot);
        }
    }
}

void MotionPlanner::setAlgorithm(BaseAlgorithm *algorithm) {
    this->algorithm = algorithm;
    this->cudaAlgorithm = nullptr;
    this->useCUDA = false;
}

void MotionPlanner::setAlgorithm(BaseAlgorithmCUDA *cudaAlgorithm) {
    this->cudaAlgorithm = cudaAlgorithm;
    this->algorithm = nullptr;
    this->useCUDA = true;
}
