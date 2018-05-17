#ifndef PIPE_H
#define PIPE_H

#include "inferxlite_common.h"
#include "hash.h"
#ifdef __cplusplus
extern "C"
{
#endif

extern void pipeline_init(struct handler *hd);
extern void weight_bias_rand_init(struct model_pipeline* m, int nw, struct handler *hd);
extern void input_rand_init(struct data_pipeline* input, struct handler *hd);
extern struct model_pipeline* get_model(char *name, int nw, int *nv, int *vw, struct data_arg dg, struct handler *hd);
extern void load_model_and_data_from_ball(char *fname, struct handler *hd);
extern void load_model_from_binary(char *fname_pre, struct handler *hd);
extern void print_time_by_layer(struct handler *hd);
extern void check_size(struct model_pipeline* model, int nw, int *nv, int *vw, char *name);
extern void handler_init(inferx_context *ctx);
extern void func_pointer_init(char *path, struct handler *hd);
extern void inferx_keep_data_shape(struct data_pipeline *input, struct data_pipeline *output);
extern void inferx_update_data_shape(struct data_pipeline *input, struct data_pipeline *output, int output_c, int pad_h, int pad_w, int kernel_h, int kernel_w, int str_h, int str_w, int dila_h, int dila_w, char func);
extern int inferx_get_data_len(DLTensor input, int *nchw, int output_c, int pad_h, int pad_w, int kernel_h, int kernel_w, int str_h, int str_w, char func);
extern void inferx_update_input_data(char *bottom, struct handler *hd);
extern void inferx_update_layer_timer(struct timeval end_layer, char *mname, struct handler *hd);
extern void inferx_insert_model_func(char *name, model_func_pointer func, struct handler *hd);
extern void inferx_str_to_int(const char *nchw_c, int *nchw_l);
extern void inferx_parse_str(char *data, int *nchw);
extern void inferx_set_init_var(bool *weight_has_load, bool *data_has_init, char *weight, char *data, struct handler *hd);
extern bool inferx_var_add_init(char *var, struct handler *hd);

#ifdef __cplusplus
}
#endif //cpp
#endif //PIPE_H
