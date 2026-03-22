/**
 * feature_detect.cpp
 * Shivang Patel (shivang2402)
 * Project 4: Task 7 - Robust Feature Detection
 *
 * Detects Harris corners and ORB features on images.
 * Keys:
 *   h = Harris corners
 *   o = ORB features
 *   +/- = adjust threshold/number of features
 *   ] / [ = next/prev image
 *   z = save screenshot
 *   q = quit
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

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
            std::cerr << "Cannot open camera." << std::endl;
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

    cv::namedWindow("Feature Detection", cv::WINDOW_AUTOSIZE);

    char mode = 'h'; // h = Harris, o = ORB
    int harrisBlockSize = 2;
    int harrisKsize = 3;
    double harrisK = 0.04;
    double harrisThreshold = 150.0;
    int orbMaxFeatures = 500;
    int screenshotCount = 0;

    std::cout << "Keys: h=Harris o=ORB +/-=threshold z=screenshot q=quit" << std::endl;

    cv::Mat frame, display;

    for (;;) {
        if (!imageFiles.empty()) {
            frame = cv::imread(imageFiles[imgIndex]);
            if (frame.empty()) break;
        } else {
            cap >> frame;
            if (frame.empty()) break;
        }

        // Resize large images for performance
        cv::Mat resized;
        if (frame.cols > 1280) {
            double scale = 1280.0 / frame.cols;
            cv::resize(frame, resized, cv::Size(), scale, scale);
        } else {
            resized = frame;
        }

        display = resized.clone();
        cv::Mat gray;
        cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);

        if (mode == 'h') {
            // Harris Corner Detection
            cv::Mat harrisResponse;
            cv::cornerHarris(gray, harrisResponse, harrisBlockSize, harrisKsize, harrisK);

            // Normalize to 0-255
            cv::Mat harrisNorm;
            cv::normalize(harrisResponse, harrisNorm, 0, 255, cv::NORM_MINMAX, CV_32FC1);

            // Draw circles at corners above threshold
            int count = 0;
            for (int r = 0; r < harrisNorm.rows; r++) {
                for (int c = 0; c < harrisNorm.cols; c++) {
                    if (harrisNorm.at<float>(r, c) > harrisThreshold) {
                        cv::circle(display, cv::Point(c, r), 5, cv::Scalar(0, 0, 255), 2);
                        count++;
                    }
                }
            }

            std::string info = "Harris | Threshold: " + std::to_string((int)harrisThreshold) + " | Features: " + std::to_string(count);
            cv::putText(display, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        } else if (mode == 'o') {
            // ORB Feature Detection
            cv::Ptr<cv::ORB> orb = cv::ORB::create(orbMaxFeatures);
            std::vector<cv::KeyPoint> keypoints;
            orb->detect(gray, keypoints);

            cv::drawKeypoints(display, keypoints, display, cv::Scalar(0, 255, 0),
                              cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

            std::string info = "ORB | Max: " + std::to_string(orbMaxFeatures) + " | Detected: " + std::to_string((int)keypoints.size());
            cv::putText(display, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("Feature Detection", display);

        char key = (char)cv::waitKey(useCamera ? 10 : 0);

        if (key == 'h') {
            mode = 'h';
            std::cout << "Mode: Harris Corners" << std::endl;
        }
        if (key == 'o') {
            mode = 'o';
            std::cout << "Mode: ORB Features" << std::endl;
        }
        if (key == '+' || key == '=') {
            if (mode == 'h') {
                harrisThreshold += 10;
                std::cout << "Harris threshold: " << harrisThreshold << std::endl;
            } else {
                orbMaxFeatures += 100;
                std::cout << "ORB max features: " << orbMaxFeatures << std::endl;
            }
        }
        if (key == '-') {
            if (mode == 'h') {
                harrisThreshold = std::max(10.0, harrisThreshold - 10);
                std::cout << "Harris threshold: " << harrisThreshold << std::endl;
            } else {
                orbMaxFeatures = std::max(100, orbMaxFeatures - 100);
                std::cout << "ORB max features: " << orbMaxFeatures << std::endl;
            }
        }
        if (key == ']' && !imageFiles.empty()) {
            imgIndex = (imgIndex + 1) % (int)imageFiles.size();
            std::cout << "Image: " << imageFiles[imgIndex] << std::endl;
        }
        if (key == '[' && !imageFiles.empty()) {
            imgIndex = (imgIndex - 1 + (int)imageFiles.size()) % (int)imageFiles.size();
            std::cout << "Image: " << imageFiles[imgIndex] << std::endl;
        }
        if (key == 'z' || key == 'Z') {
            std::string fname = "data/reports/feature_" + std::to_string(screenshotCount++) + ".png";
            cv::imwrite(fname, display);
            std::cout << "Saved: " << fname << std::endl;
        }
        if (key == 'q' || key == 'Q') break;
    }

    cv::destroyAllWindows();
    return 0;
}