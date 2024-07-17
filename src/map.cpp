#include "motion-planning/map.h"

Map::Map(int width, int height) : width(width), height(height), grid(width, std::vector<bool>(height, false)) {}
Map::~Map() {}

// methods
void Map::addObstacle(int x, int y) {
    if (x >= 0 && x < width && y >= 0 && y < height) {
        grid[x][y] = true;
    }
}
void Map::removeObstacle(int x, int y) {
    if (x >= 0 && x < width && y >= 0 && y < height) {
        grid[x][y] = false;
    }
}
bool Map::isOccupied(int x, int y) const {
    if (x >= 0 && x < width && y >= 0 && y < height) {
        return grid[x][y];
    }
    return true;
}

int Map::getWidth() const {
    return width;
}

int Map::getHeight() const {
    return height;
}

void Map::drawMap(cv::Mat &image) {
    image = cv::Mat::zeros(height, width, CV_8UC3);
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            if (grid[x][y]) {
                image.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255); // Red for obstacles
            }
        }
    }
}
