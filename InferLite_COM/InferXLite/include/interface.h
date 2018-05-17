#ifndef INTERFACE_H
#define INTERFACE_H
#include "inferxlite_common.h"
#ifdef __cplusplus
extern "C"
{
#endif //cpp

extern void inferx_convolution(int input_c, int output_c, int kernel_h, int kernel_w, int str_h, int str_w, int pad_h, int pad_w, int group, int dilation, int axis, bool bias, bool force_nd_im2col, char *bottom_pre, char *top_pre, char *iname_pre, char *model_pre, char *data_pre, int activation_type, struct handler *hd);
extern void inferx_deconvolution(int input_c, int output_c, int kernel_h, int kernel_w, int str_h, int str_w, int pad_h, int pad_w, int group, int dilation, int axis, bool bias, bool force_nd_im2col, char *bottom_pre, char *top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd);
extern void inferx_global_pooling(enum OPERATION op, char *bottom_pre, char *top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd);
extern void inferx_inner_product(int num_input, int num_output, bool bias, bool transpose, char *bottom_pre, char *top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd);
extern void inferx_relu(char* bottom_pre, char* top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd);
extern void inferx_power(float power, float scale, float shift, char* bottom_pre, char* top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd);
extern void inferx_tanh(char* bottom_pre, char* top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd);
extern void inferx_sigmoid(char* bottom_pre, char* top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd);
extern void inferx_log_softmax(char* bottom_pre, char* top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd);
extern void inferx_softmax(int axis, char* bottom_pre, char* top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd);
extern void inferx_lrn(int local_size, float alpha, float beta, char *bottom_pre, char *top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd);
extern void inferx_batchnorm(float moving_average_fraction, float eps, bool use_global_stats, char *bottom_pre,char *top_pre,char *iname_pre, char *model_pre, char *data_pre, struct handler *hd);
extern void inferx_scale(int axis,int num_axes, bool bias, char* bottom_pre, char* top_pre, char* iname_pre, char *model_pre, char *data_pre, struct handler *hd);
extern void inferx_input(int *nchw, void *pdata, char* top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd);
extern void inferx_print_data(char *bottom_pre, char *data_pre, struct handler *hd);
extern void inferx_sort_data(char *bottom_pre, char *data_pre, struct handler *hd);
extern void inferx_save_data(char *path, char *bottom_pre, char *data_pre, struct handler *hd);
extern void inferx_zero_data(struct data_pipeline *p);
extern void inferx_concat(int num_output,int axis,int concat_dim,int bottom_num,char **bottoms_pre,char *top_pre,char *iname_pre, char *model_pre, char *data_pre, struct handler *hd);
extern void inferx_slice(int axis, char *bottom_pre, char *top1_pre, char *top2_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd);
extern void inferx_eltwise(int coeffs_num, float* coeffs, enum OPERATION op, bool stabel_prod_grad, int bottom_num, char **bottoms_pre, char *top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd);
extern void inferx_elem_wise_operate(int coeffs_num, float* coeffs,enum OPERATION op,bool stabel_prod_grad,int bottom_num,char **bottoms_pre,char *top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd);
extern void inferx_crop(int axis,int offset,char* bottom_pre, char* bottom_mode_pre,char *top_pre,char *iname_pre, char *model_pre, char *data_pre, struct handler *hd);
extern void inferx_finalize(char* modelname, struct handler *hd);
extern void inferx_reshape(int n, int c, int h, int w, char *bottom_pre, char* top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd);
extern void inferx_reorg(int n, int c, int h, int w, char *bottom_pre, char* top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd);
extern void* inferx_get_data(char *bottom_pre, char *data_pre, struct handler *hd);
extern void* inferx_get_model(char *bottom_pre, char *data_pre, struct handler *hd);
extern void inferx_pooling_yolo(int kernel_h, int kernel_w, int str_h, int str_w, int pad_h, int pad_w,enum OPERATION pool, char *bottom_pre, char *top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd);
extern void inferx_pooling(int kernel_h, int kernel_w, int str_h, int str_w, int pad_h, int pad_w,enum OPERATION pool, char *bottom_pre, char *top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd);
extern void inferx_net_preprocess(char *data_c, char *model, int nchw[4], struct handler *hd);

#ifdef __cplusplus
}
#endif//cpp

#endif// INTERFACE_H
