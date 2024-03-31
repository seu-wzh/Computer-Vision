#include <iostream>
#include <yaml-cpp/yaml.h>
#include "canny.hpp"

int main(int argc, char** argv) {
    if (argc != 2 and argc != 3) {
        std::cerr << "argument error" << std::endl;
        exit(EXIT_FAILURE);
    }

    YAML::Node config {YAML::LoadFile(argv[1])};
    int median_size    = 2 * config["median"]["radius"].as<int>() + 1;
    int bilateral_size = 2 * config["bilateral"]["radius"].as<int>() + 1;
    double sigma_r = config["bilateral"]["sigma_r"].as<double>();
    double sigma_s = config["bilateral"]["sigma_s"].as<double>();
    double max     = config["threshold"]["max"].as<double>();
    double min     = config["threshold"]["min"].as<double>();

    cv::Mat image, gray, median, bilateral;

#ifdef CUSTOMIZE
    std::CannyDetector::detection_type detection;
#else
    cv::Mat detection;
#endif

    auto loop = [&] (cv::VideoCapture capture, int delay) {
        cv::namedWindow("camera", cv::WINDOW_NORMAL);
        cv::resizeWindow("camera", 640*2, 480*2);
        capture >> image;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cv::imshow("camera", gray);

        cv::medianBlur(gray, median, median_size);
        cv::namedWindow("median", cv::WINDOW_NORMAL);
        cv::resizeWindow("median", 640*2, 480*2);
        cv::imshow("median", median);

#ifdef CUSTOMIZE
        std::BilateralFilter filter {sigma_r, sigma_s, (bilateral_size - 1) / 2};
        bilateral = filter.filtering(median);
#else
        cv::bilateralFilter(median, bilateral, bilateral_size, sigma_s, sigma_r);
#endif

        cv::Mat x_grad, y_grad;
        cv::Sobel(bilateral, x_grad, -1, 1, 0);
        cv::Sobel(bilateral, y_grad, -1, 0, 1);

        bilateral.convertTo(bilateral, CV_8U);
        cv::namedWindow("bilateral", cv::WINDOW_NORMAL);
        cv::resizeWindow("bilateral", 640*2, 480*2);
        cv::imshow("bilateral", bilateral);

#ifdef CUSTOMIZE
        x_grad.convertTo(x_grad, CV_32F);
        y_grad.convertTo(y_grad, CV_32F);
        std::CannyDetector detector;
        detection = detector.detect(x_grad, y_grad, max, min);

        cv::namedWindow("gradient", cv::WINDOW_NORMAL);
        cv::resizeWindow("gradient", 640*2, 480*2);
        cv::imshow("gradient", std::get<0>(detection));

        cv::namedWindow("NMS", cv::WINDOW_NORMAL);
        cv::resizeWindow("NMS", 640*2, 480*2);
        cv::imshow("NMS", std::get<1>(detection));

        cv::namedWindow("edge", cv::WINDOW_NORMAL);
        cv::resizeWindow("edge", 640*2, 480*2);
        cv::imshow("edge", std::get<2>(detection));
#else
        x_grad.convertTo(x_grad, CV_16SC1);
        y_grad.convertTo(y_grad, CV_16SC1);
        cv::Canny(x_grad, y_grad, detection, max, min);
        cv::namedWindow("edge", cv::WINDOW_NORMAL);
        cv::resizeWindow("edge", 640*2, 480*2);
        cv::imshow("edge", detection);
#endif

        return char(cv::waitKey(delay));
    };

    cv::VideoCapture capture {config["device"].as<std::string>()};
    if (!capture.isOpened()) {
        std::cerr << "failed to open the device" << std::endl;
        exit(EXIT_FAILURE);
    }

    while (true) if (loop(capture, 1) == 'q') break;

    cv::imwrite(config["image_path"]["gray"].as<std::string>(),      gray);
    cv::imwrite(config["image_path"]["median"].as<std::string>(),    median);
    cv::imwrite(config["image_path"]["bilateral"].as<std::string>(), bilateral);
#ifdef CUSTOMIZE
    cv::imwrite(config["image_path"]["gradient"].as<std::string>(), std::get<0>(detection));
    cv::imwrite(config["image_path"]["nms"].as<std::string>(),      std::get<1>(detection));
    cv::imwrite(config["image_path"]["edge"].as<std::string>(),     std::get<2>(detection));
#else
    cv::imwrite(config["image_path"]["edge"].as<std::string>(), detection);
#endif

    exit(EXIT_SUCCESS);
}