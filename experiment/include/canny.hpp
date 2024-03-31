#pragma once

#include <tuple>
#include <opencv2/opencv.hpp>
#include "bilateral.hpp"

namespace std{

class CannyDetector {

public:

    using detection_type = tuple<cv::Mat, cv::Mat, cv::Mat>;

    detection_type detect(cv::Mat x_grad, cv::Mat y_grad, float max, float min);

};

}