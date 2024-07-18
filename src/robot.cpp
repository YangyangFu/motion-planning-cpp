#include <opencv2/opencv.hpp>
#include "motion-planning/robot.h"

// constructor
Robot::Robot(const Pose &pose, const Extent &extent) : pose(pose), extent(extent) {
    updateBox();
}

// destructor
Robot::~Robot() {}

// setters
void Robot::setPose(const double x, const double y, const double theta) {
    this->pose = Pose{x, y, theta};
    updateBox();
}
void Robot::setExtent(const double x, const double y) {
    this->extent = Extent{x, y};
    updateBox();
}
void Robot::setMaxSpeed(const double speed) {
    this->maxSpeed = speed;
}
void Robot::setMaxSteering(const double steering) {
    this->maxSteering = steering;
}

// getters
Pose Robot::getPose() const {
    return pose;
}
Extent Robot::getExtent() const {
    return extent;
}
cv::RotatedRect Robot::getBox() const {
    return box;
}

double Robot::getMaxSpeed() const {
    return maxSpeed;
}

double Robot::getMaxSteering() const {
    return maxSteering;
}

// methods
void Robot::move(double linear, double angular) {
    pose.theta += angular;
    pose.x += linear * cos(pose.theta);
    pose.y += linear * sin(pose.theta);
    updateBox();
}
void Robot::drawRobot(cv::Mat &image) const {
    cv::Point2f vertices[4];
    box.points(vertices);
    for (int i = 0; i < 4; i++) {
        cv::line(image, vertices[i], vertices[(i+1)%4], cv::Scalar(0, 255, 0), 2);
    }
}
void Robot::updateBox() {
    // rotate the box based on heading
    cv::Point2f center(pose.x, pose.y);
    cv::Size2f size(2*extent.x, 2*extent.y);
    double angle = pose.theta * 180 / M_PI; // in degrees

    box = cv::RotatedRect(center, size, angle);
}