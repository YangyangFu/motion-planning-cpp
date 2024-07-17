#ifndef BASE_ALGORITHM_CUDA_H
#define BASE_ALGORITHM_CUDA_H

#include <vector>
#include <utility>
#include "motion-planning/map.h"
#include "motion-planning/robot.h"

class BaseAlgorithmCUDA {
public:
    virtual ~BaseAlgorithmCUDA() = default;
    virtual std::vector<std::pair<int, int>> findPath(const Map &map, const Robot &robot) = 0;
};

#endif // BASE_ALGORITHM_CUDA_H
