#define _USE_MATH_DEFINES
#include <math.h>
#include "canny.hpp"

namespace std {

CannyDetector::detection_type CannyDetector::detect(cv::Mat x_grad, cv::Mat y_grad, float max, float min) {
    cv::Mat gradient {x_grad.mul(x_grad) + y_grad.mul(y_grad)};
    cv::normalize(gradient, gradient, 0., 255., cv::NORM_MINMAX);
    cv::Mat nms {cv::Mat::zeros(x_grad.rows, x_grad.cols, CV_32F)};

    /* -------------------------------- gradient -------------------------------- */

    float  grad, dx, dy, dir, grad_a, grad_b, alpha;
    float* grad_ptr   = (float*)gradient.data;
    float* nms_ptr    = (float*)nms.data;
    float* x_grad_ptr = (float*)x_grad.data;
    float* y_grad_ptr = (float*)y_grad.data;
    for (int i = 1; i < x_grad.rows - 1; i++) {
        for (int j = 1; j < x_grad.cols - 1; j++) {
            int pos = i * x_grad.cols + j;
            grad = grad_ptr[pos];
            dx   = x_grad_ptr[pos];
            dy   = y_grad_ptr[pos];
            dir  = atan2(dy, dx);
            if (dir < 0)
                dir = M_PI + dir;
            auto combine = [] (float m, float n, float a) { return m * a + n * (1 - a); };

            if (dir <= M_PI / 4) {
                alpha  = tan(dir);
                grad_a = combine(grad_ptr[pos + 1], grad_ptr[pos + 1 + x_grad.cols], alpha);
                grad_b = combine(grad_ptr[pos - 1], grad_ptr[pos - 1 - x_grad.cols], alpha);
            } else if (M_PI / 4 < dir and dir < M_PI / 2) {
                alpha  = 1 / tan(dir);
                grad_a = combine(grad_ptr[pos + x_grad.cols], grad_ptr[pos + x_grad.cols + 1], alpha);
                grad_b = combine(grad_ptr[pos - x_grad.cols], grad_ptr[pos - x_grad.cols - 1], alpha);
            } else if(dir == M_PI / 2) {
                grad_a = grad_ptr[pos - x_grad.cols];
                grad_b = grad_ptr[pos + x_grad.cols];
            } else if (M_PI / 2 < dir and dir <= M_PI * 3 / 4) {
                alpha  = 1 / tan(M_PI - dir);
                grad_a = combine(grad_ptr[pos + x_grad.cols], grad_ptr[pos + x_grad.cols - 1], alpha);
                grad_b = combine(grad_ptr[pos - x_grad.cols], grad_ptr[pos - x_grad.cols + 1], alpha);
            } else {
                alpha = tan(M_PI - dir);
                grad_a = combine(grad_ptr[pos + 1], grad_ptr[pos + 1 - x_grad.cols], alpha);
                grad_b = combine(grad_ptr[pos - 1], grad_ptr[pos - 1 + x_grad.cols], alpha);
            }

            if (grad > grad_a and grad > grad_b)
                nms_ptr[pos] = grad;
        }
    }

    /* ------------------------- non-maximum suppression ------------------------ */

    cv::Mat threshold;
    nms.copyTo(threshold);
    float* thres_ptr = (float*)threshold.data;
    for (int i = 0; i < x_grad.rows; i++) {
        for (int j = 0; j < x_grad.cols; j++) {
            int pos = i * x_grad.cols + j;
            grad = thres_ptr[pos];
            if (grad > max)
                thres_ptr[pos] = 255.;
            else if (grad < min)
                thres_ptr[pos] = 0.;
            else
                thres_ptr[pos] = 128.;
        }
    }
    auto link = [] (int anchor_pos, int cols, float* ptr) {
        for (int k = -1; k <= 1; k++) {
            for (int l = -1; l <= 1; l++) {
                if (ptr[anchor_pos + k * cols + l] == 255.) {
                    ptr[anchor_pos] = 255.;
                    return;
                }
            }
        }
        ptr[anchor_pos] = 0.;
    };
    for (int i = 1; i < x_grad.rows - 1; i++) {
        for (int j = 1; j < x_grad.cols - 1; j++) {
            int pos = i * x_grad.cols + j;
            grad = thres_ptr[pos];
            if (grad == 128.)
                link(pos, x_grad.cols, thres_ptr);
        }
    }

    /* ---------------------------- double threshold ---------------------------- */

    gradient.convertTo(gradient, CV_8U);
    nms.convertTo(nms, CV_8U);
    threshold.convertTo(threshold, CV_8U);
    return {gradient, nms, threshold};
}

}