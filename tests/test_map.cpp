/* add some test for map */

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "motion-planning/map.h"

TEST(MapTest, AddObstacle) {
    Map map(10, 10);
    map.addObstacle(5, 5);
    EXPECT_TRUE(map.isOccupied(5, 5));
};

TEST(MapTest, RemoveObstacle) {
    Map map(10, 10);
    map.addObstacle(5, 5);
    map.removeObstacle(5, 5);
    EXPECT_FALSE(map.isOccupied(5, 5));
};

TEST(MapTest, IsOccupied) {
    Map map(10, 10);
    map.addObstacle(5, 5);
    EXPECT_TRUE(map.isOccupied(5, 5));
    EXPECT_FALSE(map.isOccupied(0, 0));
};

TEST(MapTest, GetWidth) {
    Map map(10, 10);
    EXPECT_EQ(map.getWidth(), 10);
};

TEST(MapTest, GetHeight) {
    Map map(10, 10);
    EXPECT_EQ(map.getHeight(), 10);
};

TEST(MapTest, DrawMap) {
    Map map(10, 10);
    map.addObstacle(5, 5);
    cv::Mat image;
    map.drawMap(image);

    // save img to local
    //cv::imwrite("test_map.png", image);

    EXPECT_EQ(image.at<cv::Vec3b>(5, 5), cv::Vec3b(0, 0, 255));
    EXPECT_EQ(image.at<cv::Vec3b>(0, 0), cv::Vec3b(0, 0, 0));
};

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
