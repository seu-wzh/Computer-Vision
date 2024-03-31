#include <math.h>
#include "bilateral.hpp"

namespace std {

cv::Mat BilateralFilter::filtering(cv::Mat src) {
    cv::Mat pad_src;
    cv::copyMakeBorder(
        /* src & dst    */  src, pad_src, 
        /* top          */  this->radius, 
        /* bottom       */  this->radius, 
        /* left         */  this->radius, 
        /* right        */  this->radius, 
        /* type & const */  cv::BORDER_CONSTANT, 0
    );
    pad_src.convertTo(pad_src, CV_32F);
    cv::Mat dst(src.rows, src.cols, CV_32F);
    float  spatial, range, weight, weight_sum;
    float* src_ptr = (float*)pad_src.data;
    float* dst_ptr = (float*)dst.data;
    for (int i = this->radius; i < src.rows + this->radius; i++) {
        for (int j = this->radius; j < src.cols + this->radius; j++) {
            int dst_pos = (i - this->radius) * src.cols + (j - this->radius);
            dst_ptr[dst_pos] = 0;
            weight_sum = 0;
            int src_pos_a = i * (src.cols + 2 * this->radius) + j, src_pos_b;
            for (int k = -this->radius; k <= this->radius; k++) {
                for (int l = -this->radius; l <= this->radius; l++) {
                    spatial   = (k * k + l * l) / (2 * this->sigma_s * this->sigma_s);
                    src_pos_b = (i + k) * (src.cols + 2 * this->radius) + (j + l);
                    range     = src_ptr[src_pos_b] - src_ptr[src_pos_a];
                    range     = (range * range) / (2 * this->sigma_r * this->sigma_r);
                    weight    = exp(-(spatial + range));
                    dst_ptr[dst_pos] += weight * src_ptr[src_pos_b];
                    weight_sum += weight;
                }
            }
            if (weight_sum != 0)
                dst_ptr[dst_pos] /= weight_sum;
        }
    }
    cv::normalize(dst, dst, 0., 255., cv::NORM_MINMAX);
    return dst;
}

}