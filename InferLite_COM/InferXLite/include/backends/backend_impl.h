#ifndef BACKEND_IMPL_H_
#define BACKEND_IMPL_H_

#include "inferxlite_common.h"
#ifdef __cplusplus
extern "C"
{
#endif //cpp

typedef int size_int;

static inline float safe_log(float input);

extern void unpack_no_dilation_nchw_impl(struct CONV_ARG * conv_arg_pt, const float* input, float* unpack);
extern void unpack_dilation_nchw_impl(struct CONV_ARG * conv_arg_pt,const float* input, float* unpack);
extern void unpack_nhwc_impl(struct CONV_ARG * cov_arg_pt, const float* input, float* unpack);
extern void pack_nchw_impl(struct CONV_ARG * conv_arg_pt, const float* unpack, float* input);
extern void pack_nhwc_impl(struct CONV_ARG * conv_arg_pt, const float* unpack, float* input);
extern void gemm(const float* A, const float* B, float* C, bool transa, bool transb, float alpha, float beta, int m, int n, int k);
extern void quick_sort_descend(float* arr, int start, int end);
extern void convolution_forward_naive_no_pad_nchw_impl(struct CONV_ARG * conv_arg_pt, float *input, float *filter, float *output);
extern void convolution_forward_naive_pad_nchw_impl(struct CONV_ARG * conv_arg_pt, float *input, float *filter, float *output);
extern void convolution_forward_naive_pad_nhwc_impl(struct CONV_ARG * conv_arg_pt,float *input, float *filter, float *output);
extern void convolution_forward_perfdnn_nchw_impl(struct CONV_ARG * conv_arg_pt, float *input, float *filter, float *bias, float *output, bool *is_bias, int activation_type);
extern void convolution_forward_gemm_nchw_impl(struct CONV_ARG * conv_arg_pt, float *input, float *filter, float *output);
extern void convolution_backward_gemm_nchw_impl(struct CONV_ARG * conv_arg_pt,float *input, float *filter, float *output);
extern void convolution_backward_naive_no_pad_nchw_impl(struct CONV_ARG * conv_arg_pt,float *Input, float *Filter, float *Output);
extern void convolution_backward_naive_pad_nchw_impl(struct CONV_ARG * conv_arg_pt,float *input, float *filter, float *output);
extern void max_pooling_forward_nchw_impl(struct CONV_ARG * conv_arg_pt,float *input, float *output);
extern void max_pooling_forward_nchw_yolo_impl(struct CONV_ARG * conv_arg_pt,float *input, float *output);
extern void max_pooling_forward_nhwc_impl(struct CONV_ARG * conv_arg_pt,float *input, float *output);
extern void ave_pooling_forward_nchw_impl(struct CONV_ARG * conv_arg_pt,float *input, float *output);
extern void ave_pooling_forward_nhwc_impl(struct CONV_ARG * conv_arg_pt,float *input, float *output);
extern void max_pooling_forward_nc3x3s2_impl(struct CONV_ARG * conv_arg_pt,float *input, float *output);
extern void relu_forward_impl(float *input, float *output, int nchw, float slope);
extern void sigmoid_forward_impl(float *input, float *output, int nchw);
extern void softmax_forward_impl(float *input, float *output, int n, int chw);
extern void log_softmax_forward_impl(float *input, float *output, int n, int chw);
extern void bias_forward_impl(float *input, float *bias, float *output, int k, int pq);
extern void batch_norm_forward_impl(float *input, float *output, float scale_factor, float *bn_scale1, float *bn_scale2, float eps, int n, int c, int h, int w);
extern void scale_forward_impl(float *input, float *output, float *gama, float *beta, int n, int c, int h, int w, bool bias);
extern void lrn_across_forward_impl(float *input, float *output, int local_size, float alpha, float beta, int channels, int height, int width);
extern void lrn_within_forward_impl(float *input, float *output, int local_size, float alpha, float beta, int channels, int height, int width);
extern void elem_wise_operate_impl(int num_input, float **input, float *output, int len, enum OPERATION op);
extern void crop_forward_impl(float *input, float *output,  int axis, int n, int c, int h, int w, int on, int oc, int oh, int ow, int offset_n, int offset_c, int offset_h, int offset_w);
extern void tensor_concat_impl(int axis, int num_input, float **input, float *output, int *n, int *c, int *h, int *w, int no, int co, int ho, int wo);
extern void power_forward_impl(float *input, float *output, int nchw, float power, float scale, float shift);
extern void tanh_forward_impl(float *input, float *output, int nchw);
extern void reorg_forward_impl(float *x, int n, int c, int h, int w, int stride, int forward, float *out);

#ifdef __cplusplus
}
#endif //cpp
#endif //BACKEND_IMPL_H_
