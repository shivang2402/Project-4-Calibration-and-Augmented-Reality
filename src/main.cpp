/**
 * main.cpp
 * Shivang Patel (shivang2402)
 * Project 4: Calibration and Augmented Reality
 *
 * Keys:
 *   s = save current frame for calibration
 *   c = run calibration (need 5+ saved frames)
 *   w = write calibration to file
 *   l = load calibration from file
 *   p = toggle pose estimation (solvePnP)
 *   a = toggle 3D axes projection
 *   v = toggle virtual object
 *   ] / [ = next/prev image (image mode)
 *   z = save screenshot
 *   q = quit
 */

#include "calibration.h"
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    std::string inputPath = "";
    bool useCamera = true;

    if (argc >= 2) {
        inputPath = argv[1];
        useCamera = false;
    }

    cv::VideoCapture cap;
    std::vector<std::string> imageFiles;
    int imgIndex = 0;

    if (useCamera) {
        cap.open(0);
        if (!cap.isOpened()) {
            std::cerr << "Cannot open camera. Provide image path as argument." << std::endl;
            return 1;
        }
        std::cout << "Camera opened: "
                  << cap.get(cv::CAP_PROP_FRAME_WIDTH) << " x "
                  << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    } else {
        if (inputPath.find(".mp4") != std::string::npos ||
            inputPath.find(".avi") != std::string::npos ||
            inputPath.find(".mov") != std::string::npos) {
            cap.open(inputPath);
            if (!cap.isOpened()) {
                std::cerr << "Cannot open video: " << inputPath << std::endl;
                return 1;
            }
        } else {
            cv::glob(inputPath + "/*.png", imageFiles, false);
            std::vector<std::string> jpgFiles;
            cv::glob(inputPath + "/*.jpg", jpgFiles, false);
            imageFiles.insert(imageFiles.end(), jpgFiles.begin(), jpgFiles.end());
            std::sort(imageFiles.begin(), imageFiles.end());
            if (imageFiles.empty()) {
                std::cerr << "No images found in: " << inputPath << std::endl;
                return 1;
            }
            std::cout << "Loaded " << imageFiles.size() << " images from " << inputPath << std::endl;
        }
    }

    cv::namedWindow("Calibration AR", cv::WINDOW_AUTOSIZE);

    // Checkerboard: 9 columns x 6 rows of internal corners
    cv::Size patternSize(9, 6);

    // Calibration data
    std::vector<std::vector<cv::Vec3f>> point_list;
    std::vector<std::vector<cv::Point2f>> corner_list;

    // Camera intrinsics
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    bool calibrated = false;

    // Display toggles
    bool showPose = false;
    bool showAxes = false;
    bool showVirtualObj = false;
    bool showHideTarget = false;
    bool showTree = false;

    int screenshotCount = 0;
    cv::Mat frame, display;

    // Last successfully detected corners
    std::vector<cv::Point2f> lastCorners;
    bool lastFound = false;

    std::cout << "Keys: s=save c=calibrate w=write l=load p=pose a=axes v=virtual h=hide t=tree z=screenshot q=quit" << std::endl;
    if (!imageFiles.empty()) {
        std::cout << "      [/] = prev/next image" << std::endl;
    }

    for (;;) {
        // Get frame
        if (!imageFiles.empty()) {
            frame = cv::imread(imageFiles[imgIndex]);
            if (frame.empty()) break;
        } else {
            cap >> frame;
            if (frame.empty()) break;
        }

        display = frame.clone();

        // Task 1: Detect checkerboard corners every frame
        std::vector<cv::Point2f> corner_set;
        bool found = findAndDrawCorners(frame, display, patternSize, corner_set);

        if (found) {
            lastCorners = corner_set;
            lastFound = true;

            // Tasks 4/5/6: If calibrated, compute pose and project
            if (calibrated) {
                cv::Mat rvec, tvec;
                std::vector<cv::Vec3f> point_set = buildPointSet(patternSize);

                bool poseOk = getPose(point_set, corner_set, camera_matrix, dist_coeffs, rvec, tvec);

                if (poseOk) {
                    if (showPose) {
                        std::cout << "Rotation: " << rvec.t() << std::endl;
                        std::cout << "Translation: " << tvec.t() << std::endl;
                    }
                    if (showHideTarget) {
                        hideTarget(display, patternSize, camera_matrix, dist_coeffs, rvec, tvec);
                    }
                    if (showAxes) {
                        drawAxes(display, camera_matrix, dist_coeffs, rvec, tvec);
                    }
                    if (showVirtualObj) {
                        drawVirtualObject(display, camera_matrix, dist_coeffs, rvec, tvec);
                    }
                    if (showTree) {
                        drawTree(display, camera_matrix, dist_coeffs, rvec, tvec);
                    }
                }
            }
        }

        // Show info overlay
        std::string info = "Saved: " + std::to_string(corner_list.size());
        if (calibrated) info += " | CALIBRATED";
        cv::putText(display, info, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Calibration AR", display);

        char key = (char)cv::waitKey(useCamera ? 10 : 0);

        // Task 2: Save calibration frame
        if (key == 's' || key == 'S') {
            if (lastFound) {
                std::vector<cv::Vec3f> point_set = buildPointSet(patternSize);
                corner_list.push_back(lastCorners);
                point_list.push_back(point_set);

                // Save the calibration image
                std::string fname = "data/calibration_images/calib_" + std::to_string(corner_list.size()) + ".png";
                cv::imwrite(fname, frame);
                std::cout << "Saved calibration frame " << corner_list.size() << " (" << fname << ")" << std::endl;
            } else {
                std::cout << "No checkerboard detected in current frame. Cannot save." << std::endl;
            }
        }

        // Task 3: Run calibration
        if (key == 'c' || key == 'C') {
            if (corner_list.size() < 5) {
                std::cout << "Need at least 5 calibration frames. Currently have: " << corner_list.size() << std::endl;
            } else {
                double error = runCalibration(corner_list, point_list, frame.size(), camera_matrix, dist_coeffs);
                calibrated = true;
                std::cout << "Calibration complete! Reprojection error: " << error << std::endl;
                std::cout << "Camera matrix:\n" << camera_matrix << std::endl;
                std::cout << "Distortion coefficients: " << dist_coeffs << std::endl;
            }
        }

        // Write calibration to file
        if (key == 'w' || key == 'W') {
            if (calibrated) {
                writeCalibration("data/calibration.yml", camera_matrix, dist_coeffs);
                std::cout << "Calibration written to data/calibration.yml" << std::endl;
            } else {
                std::cout << "Not calibrated yet." << std::endl;
            }
        }

        // Load calibration from file
        if (key == 'l' || key == 'L') {
            if (readCalibration("data/calibration.yml", camera_matrix, dist_coeffs)) {
                calibrated = true;
                std::cout << "Calibration loaded from data/calibration.yml" << std::endl;
                std::cout << "Camera matrix:\n" << camera_matrix << std::endl;
                std::cout << "Distortion coefficients: " << dist_coeffs << std::endl;
            } else {
                std::cout << "Failed to load calibration." << std::endl;
            }
        }

        // Toggle pose printing
        if (key == 'p' || key == 'P') {
            showPose = !showPose;
            std::cout << "Pose display: " << (showPose ? "ON" : "OFF") << std::endl;
        }

        // Toggle 3D axes
        if (key == 'a' || key == 'A') {
            showAxes = !showAxes;
            std::cout << "Axes display: " << (showAxes ? "ON" : "OFF") << std::endl;
        }

        // Toggle virtual object
        if (key == 'v' || key == 'V') {
            showVirtualObj = !showVirtualObj;
            std::cout << "Virtual object: " << (showVirtualObj ? "ON" : "OFF") << std::endl;
        }

        // Toggle hide target (extension)
        if (key == 'h' || key == 'H') {
            showHideTarget = !showHideTarget;
            std::cout << "Hide target: " << (showHideTarget ? "ON" : "OFF") << std::endl;
        }

        // Toggle tree (extension)
        if (key == 't' || key == 'T') {
            showTree = !showTree;
            std::cout << "Tree: " << (showTree ? "ON" : "OFF") << std::endl;
        }

        // Screenshot
        if (key == 'z' || key == 'Z') {
            std::string fname = "data/reports/screenshot_" + std::to_string(screenshotCount++) + ".png";
            cv::imwrite(fname, display);
            std::cout << "Saved: " << fname << std::endl;
        }

        // Image navigation
        if (key == ']' && !imageFiles.empty()) {
            imgIndex = (imgIndex + 1) % (int)imageFiles.size();
            std::cout << "Image: " << imageFiles[imgIndex] << std::endl;
        }
        if (key == '[' && !imageFiles.empty()) {
            imgIndex = (imgIndex - 1 + (int)imageFiles.size()) % (int)imageFiles.size();
            std::cout << "Image: " << imageFiles[imgIndex] << std::endl;
        }

        if (key == 'q' || key == 'Q') break;
    }

    cv::destroyAllWindows();
    return 0;
}