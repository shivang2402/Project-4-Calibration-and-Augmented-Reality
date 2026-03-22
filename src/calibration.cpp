/**
 * calibration.cpp
 * Shivang Patel (shivang2402)
 * Project 4: Calibration and Augmented Reality
 */

#include "calibration.h"
#include <iostream>

// Task 1: Detect checkerboard corners with subpixel refinement
bool findAndDrawCorners(const cv::Mat &frame, cv::Mat &display,
                        cv::Size patternSize, std::vector<cv::Point2f> &corner_set) {
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    bool found = cv::findChessboardCorners(gray, patternSize, corner_set,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);

    if (found) {
        // Subpixel refinement
        cv::cornerSubPix(gray, corner_set, cv::Size(11, 11), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));

        // Draw corners on display image
        cv::drawChessboardCorners(display, patternSize, corner_set, found);

        // Print first corner info
        std::cout << "Found " << corner_set.size() << " corners. "
                  << "First: (" << corner_set[0].x << ", " << corner_set[0].y << ")" << std::endl;
    }

    return found;
}

// Task 2: Build 3D world coordinates for checkerboard corners
// Origin at upper left, X goes right, Y goes down (negative), Z = 0
std::vector<cv::Vec3f> buildPointSet(cv::Size patternSize) {
    std::vector<cv::Vec3f> point_set;
    for (int row = 0; row < patternSize.height; row++) {
        for (int col = 0; col < patternSize.width; col++) {
            point_set.push_back(cv::Vec3f(col, -row, 0));
        }
    }
    return point_set;
}

// Task 3: Camera calibration
double runCalibration(const std::vector<std::vector<cv::Point2f>> &corner_list,
                      const std::vector<std::vector<cv::Vec3f>> &point_list,
                      cv::Size imageSize,
                      cv::Mat &camera_matrix, cv::Mat &dist_coeffs) {
    // Initialize camera matrix
    camera_matrix = cv::Mat::eye(3, 3, CV_64FC1);
    camera_matrix.at<double>(0, 2) = imageSize.width / 2.0;
    camera_matrix.at<double>(1, 2) = imageSize.height / 2.0;

    std::cout << "Camera matrix BEFORE calibration:\n" << camera_matrix << std::endl;

    // Distortion coefficients (start with 5 params for radial + tangential)
    dist_coeffs = cv::Mat::zeros(5, 1, CV_64FC1);

    std::vector<cv::Mat> rvecs, tvecs;

    double error = cv::calibrateCamera(point_list, corner_list, imageSize,
        camera_matrix, dist_coeffs, rvecs, tvecs, cv::CALIB_FIX_ASPECT_RATIO);

    std::cout << "Camera matrix AFTER calibration:\n" << camera_matrix << std::endl;
    std::cout << "Distortion coefficients: " << dist_coeffs.t() << std::endl;
    std::cout << "Reprojection error: " << error << " pixels" << std::endl;

    return error;
}

// Write calibration parameters to YAML file
void writeCalibration(const std::string &filename,
                      const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "camera_matrix" << camera_matrix;
    fs << "dist_coeffs" << dist_coeffs;
    fs.release();
}

// Read calibration parameters from YAML file
bool readCalibration(const std::string &filename,
                     cv::Mat &camera_matrix, cv::Mat &dist_coeffs) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    fs["camera_matrix"] >> camera_matrix;
    fs["dist_coeffs"] >> dist_coeffs;
    fs.release();
    return true;
}

// Task 4: Estimate board pose using solvePnP
bool getPose(const std::vector<cv::Vec3f> &point_set,
             const std::vector<cv::Point2f> &corner_set,
             const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
             cv::Mat &rvec, cv::Mat &tvec) {
    return cv::solvePnP(point_set, corner_set, camera_matrix, dist_coeffs, rvec, tvec);
}

// Task 5: Draw 3D axes at the origin (X=red, Y=green, Z=blue)
void drawAxes(cv::Mat &display,
              const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
              const cv::Mat &rvec, const cv::Mat &tvec) {
    // 3D axis endpoints: origin + 3 unit vectors
    std::vector<cv::Vec3f> axisPoints = {
        {0, 0, 0},   // origin
        {3, 0, 0},   // X axis (red)
        {0, -3, 0},  // Y axis (green, negative because Y goes up in our world)
        {0, 0, 3}    // Z axis (blue, comes toward viewer)
    };

    std::vector<cv::Point2f> imgPoints;
    cv::projectPoints(axisPoints, rvec, tvec, camera_matrix, dist_coeffs, imgPoints);

    cv::Point origin = imgPoints[0];
    // X axis = red
    cv::arrowedLine(display, origin, imgPoints[1], cv::Scalar(0, 0, 255), 3, cv::LINE_AA, 0, 0.2);
    // Y axis = green
    cv::arrowedLine(display, origin, imgPoints[2], cv::Scalar(0, 255, 0), 3, cv::LINE_AA, 0, 0.2);
    // Z axis = blue
    cv::arrowedLine(display, origin, imgPoints[3], cv::Scalar(255, 0, 0), 3, cv::LINE_AA, 0, 0.2);

    cv::putText(display, "X", imgPoints[1], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    cv::putText(display, "Y", imgPoints[2], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    cv::putText(display, "Z", imgPoints[3], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
}

// Task 6: Draw a virtual 3D object (a house shape)
// Designed in world space (units = checkerboard squares), floating above the board
void drawVirtualObject(cv::Mat &display,
                       const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
                       const cv::Mat &rvec, const cv::Mat &tvec) {
    // House: base rectangle + triangular roof + chimney
    // Base sits on the board centered around (4, -2.5), roof peaks at Z=4
    std::vector<cv::Vec3f> pts3d = {
        // Base (Z=0, on the board)
        {2, -1, 0},  // 0: front left
        {6, -1, 0},  // 1: front right
        {6, -4, 0},  // 2: back right
        {2, -4, 0},  // 3: back left

        // Top of walls (Z=2)
        {2, -1, 2},  // 4: front left top
        {6, -1, 2},  // 5: front right top
        {6, -4, 2},  // 6: back right top
        {2, -4, 2},  // 7: back left top

        // Roof peaks (Z=3.5)
        {4, -1, 3.5},  // 8: front roof peak
        {4, -4, 3.5},  // 9: back roof peak

        // Chimney (on right side of roof)
        {5, -3, 2.5},  // 10
        {5.5, -3, 2.5},// 11
        {5.5, -3, 4},  // 12
        {5, -3, 4},    // 13
        {5, -3.5, 2.5},// 14
        {5.5, -3.5, 2.5},// 15
        {5.5, -3.5, 4},// 16
        {5, -3.5, 4},  // 17

        // Door (on front wall)
        {3.5, -1, 0},  // 18: door bottom left
        {4.5, -1, 0},  // 19: door bottom right
        {4.5, -1, 1.2},// 20: door top right
        {3.5, -1, 1.2},// 21: door top left
    };

    std::vector<cv::Point2f> pts2d;
    cv::projectPoints(pts3d, rvec, tvec, camera_matrix, dist_coeffs, pts2d);

    auto line = [&](int a, int b, cv::Scalar color, int thickness = 2) {
        cv::line(display, pts2d[a], pts2d[b], color, thickness, cv::LINE_AA);
    };

    // Base edges (yellow)
    cv::Scalar yellow(0, 255, 255);
    line(0, 1, yellow); line(1, 2, yellow); line(2, 3, yellow); line(3, 0, yellow);

    // Vertical edges / walls (yellow)
    line(0, 4, yellow); line(1, 5, yellow); line(2, 6, yellow); line(3, 7, yellow);

    // Top of walls (yellow)
    line(4, 5, yellow); line(5, 6, yellow); line(6, 7, yellow); line(7, 4, yellow);

    // Roof edges (red)
    cv::Scalar red(0, 0, 255);
    line(4, 8, red); line(5, 8, red);   // front roof
    line(7, 9, red); line(6, 9, red);   // back roof
    line(8, 9, red);                     // roof ridge

    // Chimney (orange)
    cv::Scalar orange(0, 140, 255);
    line(10, 11, orange); line(11, 12, orange); line(12, 13, orange); line(13, 10, orange);
    line(14, 15, orange); line(15, 16, orange); line(16, 17, orange); line(17, 14, orange);
    line(10, 14, orange); line(11, 15, orange); line(12, 16, orange); line(13, 17, orange);

    // Door (cyan)
    cv::Scalar cyan(255, 255, 0);
    line(18, 19, cyan); line(19, 20, cyan); line(20, 21, cyan); line(21, 18, cyan);
}

// Extension: Hide the target by painting over the checkerboard with a green "grass" fill
void hideTarget(cv::Mat &display, cv::Size patternSize,
                const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
                const cv::Mat &rvec, const cv::Mat &tvec) {
    // Project the four outer corners of the full checkerboard grid
    // The checkerboard has patternSize internal corners, so the outer grid is
    // (patternSize.width+1) x (patternSize.height+1) squares
    // Outer corners in world coords:
    std::vector<cv::Vec3f> outerPts = {
        {-1, 1, 0},                                                          // top left
        {(float)patternSize.width, 1, 0},                                    // top right
        {(float)patternSize.width, -(float)patternSize.height, 0},           // bottom right
        {-1, -(float)patternSize.height, 0}                                  // bottom left
    };

    std::vector<cv::Point2f> imgPts;
    cv::projectPoints(outerPts, rvec, tvec, camera_matrix, dist_coeffs, imgPts);

    std::vector<cv::Point> poly;
    for (auto &p : imgPts) poly.push_back(cv::Point((int)p.x, (int)p.y));

    // Create a semi-transparent green overlay
    cv::Mat overlay = display.clone();
    cv::fillConvexPoly(overlay, poly, cv::Scalar(34, 139, 34)); // forest green
    cv::addWeighted(overlay, 0.7, display, 0.3, 0, display);
}

// Extension: Draw a tree next to the house
void drawTree(cv::Mat &display,
              const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
              const cv::Mat &rvec, const cv::Mat &tvec) {
    std::vector<cv::Vec3f> pts3d = {
        // Trunk (brown rectangle, X=0.5 area, left side of board)
        {0.5, -2, 0},    // 0: trunk base left
        {1.0, -2, 0},    // 1: trunk base right
        {1.0, -2, 1.5},  // 2: trunk top right
        {0.5, -2, 1.5},  // 3: trunk top left
        {0.5, -2.5, 0},  // 4: trunk base left back
        {1.0, -2.5, 0},  // 5: trunk base right back
        {1.0, -2.5, 1.5},// 6: trunk top right back
        {0.5, -2.5, 1.5},// 7: trunk top left back

        // Foliage layer 1 (wide triangle, Z=1.5 to 3)
        {-0.5, -1.5, 1.5}, // 8: bottom left front
        {2.0, -1.5, 1.5},  // 9: bottom right front
        {0.75, -1.5, 3.0}, // 10: peak front
        {-0.5, -3.0, 1.5}, // 11: bottom left back
        {2.0, -3.0, 1.5},  // 12: bottom right back
        {0.75, -3.0, 3.0}, // 13: peak back

        // Foliage layer 2 (narrower triangle, Z=2.5 to 4)
        {0.0, -1.75, 2.5}, // 14: bottom left front
        {1.5, -1.75, 2.5}, // 15: bottom right front
        {0.75, -1.75, 4.0},// 16: peak front
        {0.0, -2.75, 2.5}, // 17: bottom left back
        {1.5, -2.75, 2.5}, // 18: bottom right back
        {0.75, -2.75, 4.0},// 19: peak back
    };

    std::vector<cv::Point2f> pts2d;
    cv::projectPoints(pts3d, rvec, tvec, camera_matrix, dist_coeffs, pts2d);

    auto line = [&](int a, int b, cv::Scalar color, int thickness = 2) {
        cv::line(display, pts2d[a], pts2d[b], color, thickness, cv::LINE_AA);
    };

    // Trunk (brown)
    cv::Scalar brown(19, 69, 139);
    line(0, 1, brown); line(1, 2, brown); line(2, 3, brown); line(3, 0, brown);
    line(4, 5, brown); line(5, 6, brown); line(6, 7, brown); line(7, 4, brown);
    line(0, 4, brown); line(1, 5, brown); line(2, 6, brown); line(3, 7, brown);

    // Foliage layer 1 (dark green)
    cv::Scalar dgreen(0, 180, 0);
    line(8, 9, dgreen); line(9, 10, dgreen); line(10, 8, dgreen);
    line(11, 12, dgreen); line(12, 13, dgreen); line(13, 11, dgreen);
    line(8, 11, dgreen); line(9, 12, dgreen); line(10, 13, dgreen);

    // Foliage layer 2 (light green)
    cv::Scalar lgreen(0, 255, 0);
    line(14, 15, lgreen); line(15, 16, lgreen); line(16, 14, lgreen);
    line(17, 18, lgreen); line(18, 19, lgreen); line(19, 17, lgreen);
    line(14, 17, lgreen); line(15, 18, lgreen); line(16, 19, lgreen);
}