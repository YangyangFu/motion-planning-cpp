#ifndef BASE_ALGORITHM_H
#define BASE_ALGORITHM_H

#include <vector>
#include <utility>
#include "motion-planning/map.h"
#include "motion-planning/robot.h"

class BaseAlgorithm {
public:
    virtual ~BaseAlgorithm() = default;
    virtual std::vector<std::pair<int, int>> findPath(const Map &map, const Robot &robot) = 0;
};

#endif // BASE_ALGORITHM_H
