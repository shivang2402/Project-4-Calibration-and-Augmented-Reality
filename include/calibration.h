/**
 * calibration.h
 * Shivang Patel (shivang2402)
 * Project 4: Calibration and Augmented Reality
 */

#ifndef CALIBRATION_H
#define CALIBRATION_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// Task 1: Detect checkerboard corners and draw them
// Returns true if corners found, fills corner_set with subpixel refined corners
bool findAndDrawCorners(const cv::Mat &frame, cv::Mat &display,
                        cv::Size patternSize, std::vector<cv::Point2f> &corner_set);

// Task 2: Build 3D world point set for the checkerboard
// (0,0,0), (1,0,0), ... with Z=0, Y goes negative downward
std::vector<cv::Vec3f> buildPointSet(cv::Size patternSize);

// Task 3: Run camera calibration
// Returns reprojection error
double runCalibration(const std::vector<std::vector<cv::Point2f>> &corner_list,
                      const std::vector<std::vector<cv::Vec3f>> &point_list,
                      cv::Size imageSize,
                      cv::Mat &camera_matrix, cv::Mat &dist_coeffs);

// Task 3: Write/read calibration to/from file
void writeCalibration(const std::string &filename,
                      const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs);
bool readCalibration(const std::string &filename,
                     cv::Mat &camera_matrix, cv::Mat &dist_coeffs);

// Task 4: Get pose using solvePnP
bool getPose(const std::vector<cv::Vec3f> &point_set,
             const std::vector<cv::Point2f> &corner_set,
             const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
             cv::Mat &rvec, cv::Mat &tvec);

// Task 5: Draw 3D axes at the origin
void drawAxes(cv::Mat &display,
              const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
              const cv::Mat &rvec, const cv::Mat &tvec);

// Task 6: Draw a virtual 3D object
void drawVirtualObject(cv::Mat &display,
                       const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
                       const cv::Mat &rvec, const cv::Mat &tvec);

// Extension: Hide the target by filling checkerboard squares with a solid color
void hideTarget(cv::Mat &display, cv::Size patternSize,
                const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
                const cv::Mat &rvec, const cv::Mat &tvec);

// Extension: Draw a second virtual object (tree)
void drawTree(cv::Mat &display,
              const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
              const cv::Mat &rvec, const cv::Mat &tvec);

#endif