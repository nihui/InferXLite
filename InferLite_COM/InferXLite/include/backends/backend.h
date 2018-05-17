#ifndef BACKEND_H
#define BACKEND_H

#include "inferxlite_common.h"

#ifdef __cplusplus
extern "C"
{
#endif //cpp

extern void convolution_forward_func(struct CONV_ARG *conv_arg_pt, const DLTensor input_t, DLTensor *filter_t1, const DLTensor filter_t2, DLTensor output_t, bool *bias, int activation_type, struct handler *hd);
extern void convolution_backward_func(struct CONV_ARG *conv_arg_pt, const DLTensor output_t, const DLTensor filter_t, DLTensor input_t, struct handler *hd);
extern void pooling_forward_func(struct CONV_ARG *conv_arg_pt, const DLTensor input_t, DLTensor output_t, enum OPERATION pool, struct handler *hd);
extern void pooling_forward_func_yolo(struct CONV_ARG *conv_arg_pt, const DLTensor input_t, DLTensor output_t, enum OPERATION pool, struct handler *hd);
extern void lrn_forward_func(const DLTensor input_t, DLTensor output_t, enum LRN_WAY lrn, int local_size, float alpha, float beta, int channels, int height, int width, struct handler *hd);
extern void relu_forward_func(const DLTensor input_t, DLTensor output_t, int nchw, float slope, struct handler *hd);
extern void power_forward_func(const DLTensor input_t, DLTensor output_t, int nchw, float power, float scale, float shift, struct handler *hd);
extern void tanh_forward_func(const DLTensor input_t, DLTensor output_t, int nchw, struct handler *hd);
extern void sigmoid_forward_func(const DLTensor input_t, DLTensor output_t, int nchw, struct handler *hd);
extern void softmax_forward_func(const DLTensor input_t, DLTensor output_t, int n, int chw, struct handler *hd);
extern void log_softmax_forward_func(const DLTensor input_t, DLTensor output_t, int n, int chw, struct handler *hd);
extern void bias_forward_func(const DLTensor input_t, const DLTensor bias_t, DLTensor output_t, int n, int k, int output_hw, struct handler *hd);
extern void batch_norm_forward_func(const DLTensor input_t, DLTensor output_t, DLTensor scale_factor_t, const DLTensor bn_scale1_t, const DLTensor bn_scale2_t, float eps, int n, int c, int h, int w, struct handler *hd);
extern void scale_forward_func(const DLTensor input_t, DLTensor output_t, const DLTensor gama_t, const DLTensor beta_t, int size_n, int size_c, int size_h, int size_w, bool is_bias, struct handler *hd);
extern void matrix_multiply_func(const DLTensor left_t, const DLTensor right_t, DLTensor output_t, bool transa, bool transb, float alpha, float beta, int m_left, int n_left, int m_right, int n_right, struct handler *hd);
/*
extern float evaluate_classify_forward_func(const DLTensor output_t, const DLTensor target_t, int n, int len);
extern float evaluate_regress_forward_func(const DLTensor output, const DLTensor target, int n, int len);
extern float cross_entropy_binary_forward_func(const DLTensor input, const DLTensor target, int len);
*/
extern void elem_wise_operate_func(int num_input, DLTensor* input_t, DLTensor output_t, int len, enum OPERATION op, struct handler *hd);
extern void crop_forward_func(const DLTensor input_t, DLTensor output_t,  int axis, int input_n, int input_c, int input_h, int input_w, int output_n, int output_c, int output_h, int output_w, int offset_n, int offset_c, int offset_h, int offset_w, struct handler *hd);
extern void tensor_concat_func(int axis, int num_input, DLTensor* input_t, DLTensor output_t, struct handler *hd);
extern void reorg_forward_func(DLTensor input_t, DLTensor output_t, int n, int c, int h, int w, int stride, int forward, struct handler *hd);
extern void tensor_slice_func(int axis, DLTensor input, DLTensor output1, DLTensor output2, struct handler *hd);

#ifdef __cplusplus
}
#endif //cpp 

#endif //BACKEND_H
