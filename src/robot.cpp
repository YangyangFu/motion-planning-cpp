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
cv::Rect Robot::getBox() const {
    return box;
}

// methods
void Robot::move(double linear, double angular) {
    pose.theta += angular;
    pose.x += linear * cos(pose.theta);
    pose.y += linear * sin(pose.theta);
    updateBox();
}
void Robot::drawRobot(cv::Mat &image) const {
    cv::Point center(pose.x, pose.y);
    cv::ellipse(image, center, cv::Size(extent.x, extent.y), pose.theta * 180 / M_PI, 0, 360, cv::Scalar(0, 255, 0), 2);
    cv::line(image, center, cv::Point(center.x + extent.x * cos(pose.theta), center.y + extent.x * sin(pose.theta)), cv::Scalar(0, 255, 0), 2);
}
void Robot::updateBox() {
    // rotate the box based on heading
    std::vector<cv::Point2f> vertices(4);
    vertices[0] = cv::Point2f(-extent.x, -extent.y);
    vertices[1] = cv::Point2f(extent.x, -extent.y);
    vertices[2] = cv::Point2f(extent.x, extent.y);
    vertices[3] = cv::Point2f(-extent.x, extent.y);
    for (int i = 0; i < 4; ++i) {
        double x = vertices[i].x * cos(pose.theta) - vertices[i].y * sin(pose.theta);
        double y = vertices[i].x * sin(pose.theta) + vertices[i].y * cos(pose.theta);
        vertices[i] = cv::Point2f(x + pose.x, y + pose.y);
    }
    box = cv::boundingRect(vertices);
}