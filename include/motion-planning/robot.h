#pragma once

#include <opencv2/opencv.hpp>

/* Simple Pose in 2D */
struct Pose {
    double x;
    double y;
    double theta;
};

struct Extent {
    double x;
    double y;
};

class Robot {
public:
    Robot(const Pose &pose, const Extent &extent);
    ~Robot();

    //setters
    void setPose(const double x, const double y, const double theta);
    void setExtent(const double x, const double y);
    void setMaxSpeed(const double speed);
    void setMaxSteering(const double steering);

    //getters
    Pose getPose() const;
    Extent getExtent() const;
    cv::RotatedRect getBox() const;
    double getMaxSpeed() const;
    double getMaxSteering() const;
    
    // methods
    void move(double linear, double angular);
    void drawRobot(cv::Mat &image) const;

private:
    Pose pose;
    Extent extent;
    cv::RotatedRect box;
    double maxSpeed = 1.6; // in pixel/s
    double maxSteering = 0.5; // in radians
    void updateBox();
};