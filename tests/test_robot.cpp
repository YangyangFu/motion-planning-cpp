#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "motion-planning/robot.h"

// setPose
TEST(RobotTest, SetPose) {
    Robot robot(Pose{0, 0, 0}, Extent{1, 1});
    robot.setPose(1, 1, 1);
    EXPECT_EQ(robot.getPose().x, 1);
    EXPECT_EQ(robot.getPose().y, 1);
    EXPECT_EQ(robot.getPose().theta, 1);
};

// setExtent
TEST(RobotTest, SetExtent) {
    Robot robot(Pose{0, 0, 0}, Extent{1, 1});
    robot.setExtent(2, 2);
    EXPECT_EQ(robot.getExtent().x, 2);
    EXPECT_EQ(robot.getExtent().y, 2);
};

// setMaxSpeed
TEST(RobotTest, SetMaxSpeed) {
    Robot robot(Pose{0, 0, 0}, Extent{1, 1});
    robot.setMaxSpeed(2);
    EXPECT_EQ(robot.getMaxSpeed(), 2);
};

// setMaxSteering
TEST(RobotTest, SetMaxSteering) {
    Robot robot(Pose{0, 0, 0}, Extent{1, 1});
    robot.setMaxSteering(2);
    EXPECT_EQ(robot.getMaxSteering(), 2);
};

// getPose
TEST(RobotTest, GetPose) {
    Robot robot(Pose{1, 1, 1}, Extent{1, 1});
    EXPECT_EQ(robot.getPose().x, 1);
    EXPECT_EQ(robot.getPose().y, 1);
    EXPECT_EQ(robot.getPose().theta, 1);
};

// getExtent
TEST(RobotTest, GetExtent) {
    Robot robot(Pose{0, 0, 0}, Extent{2, 2});
    EXPECT_EQ(robot.getExtent().x, 2);
    EXPECT_EQ(robot.getExtent().y, 2);
};

// getBox: the right bound is inclusive
TEST(RobotTest, GetBox) {
    Robot robot(Pose{0, 0, 0}, Extent{1, 1});
    cv::RotatedRect box = robot.getBox();
    EXPECT_EQ(box.center.x, 0);
    EXPECT_EQ(box.center.y, 0);
    EXPECT_EQ(box.size.width, 2);
    EXPECT_EQ(box.size.height, 2);
    EXPECT_EQ(box.angle, 0);

    // points 
    cv::Point2f vertices[4];
    box.points(vertices);
    EXPECT_EQ(vertices[0].x, -1); // top-left
    EXPECT_EQ(vertices[0].y, 1); 
    EXPECT_EQ(vertices[1].x, -1); // bottom-left
    EXPECT_EQ(vertices[1].y, -1); 
    EXPECT_EQ(vertices[2].x, 1); // bottom-right
    EXPECT_EQ(vertices[2].y, -1);
    EXPECT_EQ(vertices[3].x, 1); // top-right
    EXPECT_EQ(vertices[3].y, 1);
};


// move
TEST(RobotTest, Move) {
    Robot robot(Pose{0, 0, 0}, Extent{1, 1});
    robot.move(1, 0);
    EXPECT_EQ(robot.getPose().x, 1);
    EXPECT_EQ(robot.getPose().y, 0);
    EXPECT_EQ(robot.getPose().theta, 0);
    robot.move(1, M_PI / 2);
    EXPECT_EQ(robot.getPose().x, 1);
    EXPECT_EQ(robot.getPose().y, 1);
    EXPECT_EQ(robot.getPose().theta, M_PI / 2);
};

// drawRobot
TEST(RobotTest, DrawRobot) {
    Robot robot(Pose{50, 50, 0}, Extent{1, 1});
    cv::Mat image(100, 100, CV_8UC3, cv::Scalar(255, 255, 255));
    robot.drawRobot(image);
    // save img to local
    cv::imwrite("test_robot.png", image);
    EXPECT_EQ(image.at<cv::Vec3b>(49, 51), cv::Vec3b(0, 255, 0));
    EXPECT_EQ(image.at<cv::Vec3b>(49, 49), cv::Vec3b(0, 255, 0));
    EXPECT_EQ(image.at<cv::Vec3b>(51, 49), cv::Vec3b(0, 255, 0));
    EXPECT_EQ(image.at<cv::Vec3b>(51, 51), cv::Vec3b(0, 255, 0));
};

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}





