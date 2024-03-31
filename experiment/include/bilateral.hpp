#pragma once

#include <opencv2/opencv.hpp>

namespace std
{

class BilateralFilter {

public:

    BilateralFilter(double sigma_r, double sigma_s, int radius):
        sigma_r(sigma_r), sigma_s(sigma_s), radius(radius) {}

    cv::Mat filtering(cv::Mat src);

private:

    double      sigma_r;
    double      sigma_s;
    int         radius;
};

}