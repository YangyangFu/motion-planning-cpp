#ifndef MAP_H
#define MAP_H

#include <vector>
#include <opencv2/opencv.hpp>

class Map {
public:
    Map(int width, int height);
    ~Map();

    void addObstacle(int x, int y); // Add an obstacle at pixel (x, y)
    bool isOccupied(int x, int y) const; // Check if pixel (x, y) is an obstacle: white (255, 255, 255) is free, black (0, 0, 0) is occupied
    void removeObstacle(int x, int y); // Remove an obstacle at pixel (x, y)
    int getWidth() const; // Get the width of the map
    int getHeight() const; // Get the height of the map
    void drawMap(cv::Mat &image); // Draw the map on an image

private:
    int width;
    int height;
    std::vector<std::vector<bool>> grid;
};


#endif // MAP_H