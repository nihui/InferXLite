#include "hash.h"
#include "pipe.h"
#include "backend.h"
#include "backend_impl.h"
#include <math.h>
#include <string.h>
#include "..\include\interface.h"


void inferx_convolution(int input_c, int output_c, int kernel_h, int kernel_w, int str_h, int str_w, int pad_h, int pad_w, int group, int dilation, int axis, bool bias, bool force_nd_im2col, char *bottom_pre, char *top_pre, char *iname_pre, char *model_pre, char *data_pre, int activation_type, struct handler *hd)
{
  if(hd->tsw==time_layer)
    gettimeofday(&hd->start_layer, NULL);

  char iname[1000];
  char bottom[1000];
  char top[1000];
  sprintf(iname, "%s%s", model_pre, iname_pre);
  sprintf(bottom, "%s%s", data_pre, bottom_pre);
  sprintf(top, "%s%s", data_pre, top_pre);


  struct data_pipeline* input=data_map(bottom,hd);
  struct data_pipeline* output;
  if(hd->data_has_init)
  {
    output=data_map(top,hd);
  }
  else
  {
    int nchw[4];
    inferx_get_data_len(input->datas, nchw, output_c, pad_h, pad_w, kernel_h, kernel_w, str_h, str_w, 0);
    output=data_map_init(top, nchw, hd);
    inferx_update_data_shape(input, output, output_c, pad_h, pad_w, kernel_h, kernel_w, str_h, str_w, 1, 1, 0);
  }

  inferx_zero_data(output);

  struct data_arg dg;
  dg.str_h=str_h;
  dg.str_w=str_w;
  dg.pad_h=pad_h;
  dg.pad_w=pad_w;

  struct model_pipeline* model;


  int vw[6], nv[2], nw;
  if(group==1)
  {
    vw[0]=output_c;vw[1]=input_c;vw[2]=kernel_h;vw[3]=kernel_w;
    vw[4]=output_c;
    nv[0]=4;
    nv[1]=1;
    if(bias)
      nw=2;
    else
      nw=1;
    model=get_model(iname, nw, nv, vw, dg, hd);
  }
  else
  {
    nv[0]=5;
    nv[1]=1;
    vw[0]=group;vw[1]=output_c/group;vw[2]=input_c/group;vw[3]=kernel_h;vw[4]=kernel_w;
    vw[5]=output_c;
    if(bias)
      nw=2;
    else
      nw=1;
    model=get_model(iname, nw, nv, vw, dg, hd);
  }

  struct CONV_ARG conv_arg;
  conv_arg.group=group;
  conv_arg.input_n=input->datas.shape[0];
  conv_arg.input_c=input_c;
  conv_arg.input_h=input->datas.shape[2];
  conv_arg.input_w=input->datas.shape[3];
  conv_arg.kernel_h=kernel_h;
  conv_arg.kernel_w=kernel_w;
  conv_arg.output_c=output_c;
  conv_arg.output_h=output->datas.shape[2];
  conv_arg.output_w=output->datas.shape[3];
  conv_arg.pad_h=pad_h;
  conv_arg.pad_w=pad_w;
  conv_arg.str_h=str_h;
  conv_arg.str_w=str_w;
  conv_arg.dila_h=dilation;
  conv_arg.dila_w=dilation;


  convolution_forward_func(&conv_arg, input->datas, &(model->weight[0]), model->weight[1], output->datas, &bias, activation_type, hd);


  if(bias)
    bias_forward_func(output->datas, model->weight[1], output->datas, output->datas.shape[0], output->datas.shape[1], output->datas.shape[2]*output->datas.shape[3], hd);


  if(hd->tsw==time_layer)
  {
    struct timeval end_layer;
    gettimeofday(&end_layer, NULL);
    inferx_update_layer_timer(end_layer, iname, hd);
  }

  return;
}




void inferx_deconvolution(int input_c, int output_c, int kernel_h, int kernel_w, int str_h, int str_w, int pad_h, int pad_w, int group, int dilation, int axis, bool bias, bool force_nd_im2col, char *bottom_pre, char *top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd)
{
  if(hd->tsw==time_layer)
    gettimeofday(&(hd->start_layer), NULL);

  char iname[1000];
  char bottom[1000];
  char top[1000];
  sprintf(iname, "%s%s", model_pre, iname_pre);
  sprintf(bottom, "%s%s", data_pre, bottom_pre);
  sprintf(top, "%s%s", data_pre, top_pre);

  struct data_pipeline* input=data_map(bottom, hd);
  struct data_pipeline* output;
  if(hd->data_has_init)
  {
    output=data_map(top, hd);
  }
  else
  {
    int h, w;
    h=(input->datas.shape[2]-1)*str_h+kernel_h-2*pad_h;
    w=(input->datas.shape[3]-1)*str_w+kernel_w-2*pad_w;
    int nchw[4];
    nchw[0]=input->datas.shape[0];
    nchw[1]=output_c;
    nchw[2]=h;
    nchw[3]=w;
    output=data_map_init(top, nchw, hd);
  }

  inferx_zero_data(output);

  struct model_pipeline* model;

  struct data_arg dg;
  dg.str_h=str_h;
  dg.str_w=str_w;
  dg.pad_h=pad_h;
  dg.pad_w=pad_w;

  int vw[5], nv[2], nw;
  vw[0]=output_c;vw[1]=input_c;vw[2]=kernel_h;vw[3]=kernel_w;
  vw[4]=output_c;
  nv[0]=4;
  nv[1]=1;
  if(bias)
    nw=2;
  else
    nw=1;
  model=get_model(iname, nw, nv, vw, dg, hd);


  struct CONV_ARG conv_arg;
  conv_arg.group=group;
  conv_arg.input_n=output->datas.shape[0];
  conv_arg.input_c=output_c;
  conv_arg.input_h=output->datas.shape[2];
  conv_arg.input_w=output->datas.shape[3];
  conv_arg.kernel_h=kernel_h;
  conv_arg.kernel_w=kernel_w;
  conv_arg.output_c=input_c;
  conv_arg.output_h=input->datas.shape[2];
  conv_arg.output_w=input->datas.shape[3];
  conv_arg.pad_h=pad_h;
  conv_arg.pad_w=pad_w;
  conv_arg.str_h=str_h;
  conv_arg.str_w=str_w;
  conv_arg.dila_h=dilation;
  conv_arg.dila_w=dilation;


  convolution_backward_func(&conv_arg, input->datas, model->weight[0], output->datas, hd);


  if(bias)
    bias_forward_func(output->datas, model->weight[1], output->datas, output->datas.shape[0], output->datas.shape[1], output->datas.shape[2]*output->datas.shape[3], hd);

  if(hd->tsw==time_layer)
  {
    struct timeval end_layer;
    gettimeofday(&end_layer, NULL);
    inferx_update_layer_timer(end_layer, iname, hd);
  }

  return;
}



void inferx_pooling(int kernel_h, int kernel_w, int str_h, int str_w, int pad_h, int pad_w,enum OPERATION pool, char *bottom_pre, char *top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd)
{
  if(hd->tsw==time_layer)
    gettimeofday(&(hd->start_layer), NULL);

  char iname[1000];
  char bottom[1000];
  char top[1000];
  sprintf(iname, "%s%s", model_pre, iname_pre);
  sprintf(bottom, "%s%s", data_pre, bottom_pre);
  sprintf(top, "%s%s", data_pre, top_pre);


  struct data_pipeline* input=data_map(bottom, hd);
  struct data_pipeline* output;

  if(hd->data_has_init)
  {
    output=data_map(top, hd);
  }
  else
  {
    int nchw[4];
    inferx_get_data_len(input->datas, nchw, input->datas.shape[1], pad_h, pad_w, kernel_h, kernel_w, str_h, str_w, 1);
    output=data_map_init(top, nchw, hd);
    inferx_update_data_shape(input, output, input->datas.shape[1], pad_h, pad_w, kernel_h, kernel_w, str_h, str_w, 1, 1, 1);
  }

  inferx_zero_data(output);


  struct CONV_ARG conv_arg;
  conv_arg.input_n=input->datas.shape[0];
  conv_arg.input_c=input->datas.shape[1];
  conv_arg.input_h=input->datas.shape[2];
  conv_arg.input_w=input->datas.shape[3];
  conv_arg.kernel_h=kernel_h;
  conv_arg.kernel_w=kernel_w;
  conv_arg.output_c=output->datas.shape[1];
  conv_arg.output_h=output->datas.shape[2];
  conv_arg.output_w=output->datas.shape[3];
  conv_arg.pad_h=pad_h;
  conv_arg.pad_w=pad_w;
  conv_arg.str_h=str_h;
  conv_arg.str_w=str_w;


  pooling_forward_func(&conv_arg, input->datas, output->datas, pool, hd);

  if(hd->tsw==time_layer)
  {
    struct timeval end_layer;
    gettimeofday(&end_layer, NULL);
    inferx_update_layer_timer(end_layer, iname, hd);
  }

  return;
}


void inferx_pooling_yolo(int kernel_h, int kernel_w, int str_h, int str_w, int pad_h, int pad_w,enum OPERATION pool, char *bottom_pre, char *top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd)
{
  if(hd->tsw==time_layer)
    gettimeofday(&(hd->start_layer), NULL);

  char iname[1000];
  char bottom[1000];
  char top[1000];
  sprintf(iname, "%s%s", model_pre, iname_pre);
  sprintf(bottom, "%s%s", data_pre, bottom_pre);
  sprintf(top, "%s%s", data_pre, top_pre);


  struct data_pipeline* input=data_map(bottom, hd);
  struct data_pipeline* output;

  if(hd->data_has_init)
  {
    output=data_map(top, hd);
  }
  else
  {
    int nchw[4];
    inferx_get_data_len(input->datas, nchw, input->datas.shape[1], pad_h, pad_w, kernel_h, kernel_w, str_h, str_w, 1);
    nchw[2]-=1;
    nchw[3]-=1;
    output=data_map_init(top, nchw, hd);
    inferx_update_data_shape(input, output, input->datas.shape[1], pad_h, pad_w, kernel_h, kernel_w, str_h, str_w, 1, 1, 1);
    output->datas.shape[2]-=1;
    output->datas.shape[3]-=1;
  }

  inferx_zero_data(output);

  struct CONV_ARG conv_arg;
  conv_arg.input_n=input->datas.shape[0];
  conv_arg.input_c=input->datas.shape[1];
  conv_arg.input_h=input->datas.shape[2];
  conv_arg.input_w=input->datas.shape[3];
  conv_arg.kernel_h=kernel_h;
  conv_arg.kernel_w=kernel_w;
  conv_arg.output_c=output->datas.shape[1];
  conv_arg.output_h=output->datas.shape[2];
  conv_arg.output_w=output->datas.shape[3];
  conv_arg.pad_h=pad_h;
  conv_arg.pad_w=pad_w;
  conv_arg.str_h=str_h;
  conv_arg.str_w=str_w;


  pooling_forward_func_yolo(&conv_arg, input->datas, output->datas, pool, hd);

  if(hd->tsw==time_layer)
  {
    struct timeval end_layer;
    gettimeofday(&end_layer, NULL);
    inferx_update_layer_timer(end_layer, iname, hd);
  }

  return;
}





void inferx_global_pooling(enum OPERATION pool, char *bottom_pre, char *top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd)
{
  if(hd->tsw==time_layer)
    gettimeofday(&(hd->start_layer), NULL);

  char iname[1000];
  char bottom[1000];
  char top[1000];
  sprintf(iname, "%s%s", model_pre, iname_pre);
  sprintf(bottom, "%s%s", data_pre, bottom_pre);
  sprintf(top, "%s%s", data_pre, top_pre);


  struct data_pipeline* input=data_map(bottom, hd);

  struct data_pipeline* output;

  if(hd->data_has_init)
  {
    output=data_map(top, hd);
  }
  else
  {
    int nchw[4];
    inferx_get_data_len(input->datas, nchw, input->datas.shape[1], 0, 0, input->datas.shape[2], input->datas.shape[3], 1, 1, 1);
    output=data_map_init(top, nchw, hd);
    inferx_update_data_shape(input, output, input->datas.shape[1], 0, 0, input->datas.shape[2], input->datas.shape[3], 1, 1, 1, 1, 1);
  }

  inferx_zero_data(output);


  struct CONV_ARG conv_arg;
  conv_arg.input_n=input->datas.shape[0];
  conv_arg.input_c=input->datas.shape[1];
  conv_arg.input_h=input->datas.shape[2];
  conv_arg.input_w=input->datas.shape[3];
  conv_arg.kernel_h=input->datas.shape[2];
  conv_arg.kernel_w=input->datas.shape[3];
  conv_arg.output_c=output->datas.shape[1];
  conv_arg.output_h=output->datas.shape[2];
  conv_arg.output_w=output->datas.shape[3];
  conv_arg.pad_h=0;
  conv_arg.pad_w=0;
  conv_arg.str_h=1;
  conv_arg.str_w=1;


  pooling_forward_func(&conv_arg, input->datas, output->datas, pool, hd);


  if(hd->tsw==time_layer)
  {
    struct timeval end_layer;
    gettimeofday(&end_layer, NULL);
    inferx_update_layer_timer(end_layer, iname, hd);
  }

  return;
}



void inferx_inner_product(int num_input, int num_output, bool bias, bool transpose, char *bottom_pre, char *top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd)
{
  if(hd->tsw==time_layer)
    gettimeofday(&(hd->start_layer), NULL);

  char iname[1000];
  char bottom[1000];
  char top[1000];
  sprintf(iname, "%s%s", model_pre, iname_pre);
  sprintf(bottom, "%s%s", data_pre, bottom_pre);
  sprintf(top, "%s%s", data_pre, top_pre);


  struct data_pipeline* input=data_map(bottom, hd);
  struct data_pipeline* output;

  if(hd->data_has_init)
  {
     output=data_map(top, hd);
  }
  else
  {
    int nchw[4];
    nchw[0]=1; 
    nchw[1]=num_output;
    nchw[2]=1; 
    nchw[3]=1; 
    output=data_map_init(top, nchw, hd);
  }

  inferx_zero_data(output);

  if(input->datas.shape[2]!=1)
  {
    num_input=num_input*input->datas.shape[2]*input->datas.shape[3];
  }

  struct model_pipeline* model;

  struct data_arg dg;

  int vw[3], nv[2], nw;
  nv[0]=2;nv[1]=1;
  vw[0]=num_output;vw[1]=num_input;
  vw[2]=num_output;
  if(bias)
    nw=2;
  else
    nw=1;
  model=get_model(iname, nw, nv, vw, dg, hd);

 
  matrix_multiply_func(model->weight[0], input->datas, output->datas, false, false, 1.0, 0.0, num_output, num_input, num_input, 1, hd);

  if(bias)
    bias_forward_func(output->datas, model->weight[1], output->datas, output->datas.shape[0], output->datas.shape[1], output->datas.shape[2]*output->datas.shape[3], hd);

  if(hd->tsw==time_layer)
  {
    struct timeval end_layer;
    gettimeofday(&end_layer, NULL);
    inferx_update_layer_timer(end_layer, iname, hd);
  }

  return;
}


void inferx_relu(char* bottom_pre, char* top_pre, char* iname_pre, char *model_pre, char *data_pre, struct handler *hd)
{
  if(hd->tsw==time_layer)
    gettimeofday(&(hd->start_layer), NULL);

  char iname[1000];
  char bottom[1000];
  char top[1000];
  float slope=0.0;
  sprintf(iname, "%s%s", model_pre, iname_pre);
  sprintf(bottom, "%s%s", data_pre, bottom_pre);
  sprintf(top, "%s%s", data_pre, top_pre);


  struct data_pipeline* input=data_map(bottom, hd);
  struct data_pipeline* output;
 
  if(!strcmp(bottom, top))
  {
    output=input;
  }
  else
  {
    if(hd->data_has_init)
    {
      output=data_map(top, hd);
    }
    else
    {
      int nchw[4];
      nchw[0]=input->datas.shape[0];
      nchw[1]=input->datas.shape[1];
      nchw[2]=input->datas.shape[2];
      nchw[3]=input->datas.shape[3];
      output=data_map_init(top, nchw, hd);
      inferx_keep_data_shape(input, output);
    }
  }

  relu_forward_func(input->datas, output->datas, input->datas.shape[0]*input->datas.shape[1]*input->datas.shape[2]*input->datas.shape[3], slope, hd);

  if(hd->tsw==time_layer)
  {
    struct timeval end_layer;
    gettimeofday(&end_layer, NULL);
    inferx_update_layer_timer(end_layer, iname, hd);
  }

  return;
}


void inferx_tanh(char* bottom_pre, char* top_pre, char* iname_pre, char *model_pre, char *data_pre, struct handler *hd)
{
  if(hd->tsw==time_layer)
    gettimeofday(&(hd->start_layer), NULL);

  char iname[1000];
  char bottom[1000];
  char top[1000];
  sprintf(iname, "%s%s", model_pre, iname_pre);
  sprintf(bottom, "%s%s", data_pre, bottom_pre);
  sprintf(top, "%s%s", data_pre, top_pre);


  struct data_pipeline* input=data_map(bottom, hd);
  struct data_pipeline* output;
 
  if(!strcmp(bottom, top))
  {
    output=input;
  }
  else
  {
    if(hd->data_has_init)
    {
      output=data_map(top, hd);
    }
    else
    {
      int nchw[4];
      nchw[0]=input->datas.shape[0];
      nchw[1]=input->datas.shape[1];
      nchw[2]=input->datas.shape[2];
      nchw[3]=input->datas.shape[3];
      output=data_map_init(top, nchw, hd);
      inferx_keep_data_shape(input, output);
    }
  }

  tanh_forward_func(input->datas, output->datas, input->datas.shape[0]*input->datas.shape[1]*input->datas.shape[2]*input->datas.shape[3], hd);

  if(hd->tsw==time_layer)
  {
    struct timeval end_layer;
    gettimeofday(&end_layer, NULL);
    inferx_update_layer_timer(end_layer, iname, hd);
  }

  return;
}


void inferx_power(float power, float scale, float shift, char* bottom_pre, char* top_pre, char* iname_pre, char *model_pre, char *data_pre, struct handler *hd)
{
  if(hd->tsw==time_layer)
    gettimeofday(&(hd->start_layer), NULL);

  char iname[1000];
  char bottom[1000];
  char top[1000];
  sprintf(iname, "%s%s", model_pre, iname_pre);
  sprintf(bottom, "%s%s", data_pre, bottom_pre);
  sprintf(top, "%s%s", data_pre, top_pre);


  struct data_pipeline* input=data_map(bottom, hd);
  struct data_pipeline* output;
 
  if(!strcmp(bottom, top))
  {
    output=input;
  }
  else
  {
    if(hd->data_has_init)
    {
      output=data_map(top, hd);
    }
    else
    {
      int nchw[4];
      nchw[0]=input->datas.shape[0];
      nchw[1]=input->datas.shape[1];
      nchw[2]=input->datas.shape[2];
      nchw[3]=input->datas.shape[3];
      output=data_map_init(top, nchw, hd);
      inferx_keep_data_shape(input, output);
    }
  }

  power_forward_func(input->datas, output->datas, input->datas.shape[0]*input->datas.shape[1]*input->datas.shape[2]*input->datas.shape[3], power, scale, shift, hd);


  if(hd->tsw==time_layer)
  {
    struct timeval end_layer;
    gettimeofday(&end_layer, NULL);
    inferx_update_layer_timer(end_layer, iname, hd);
  }

  return;
}



void inferx_batchnorm(float moving_average_fraction, float eps, bool use_global_stats, char *bottom_pre, char *top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd)
{
  if(hd->tsw==time_layer)
    gettimeofday(&(hd->start_layer), NULL);

  char iname[1000];
  char bottom[1000];
  char top[1000];
  sprintf(iname, "%s%s", model_pre, iname_pre);
  sprintf(bottom, "%s%s", data_pre, bottom_pre);
  sprintf(top, "%s%s", data_pre, top_pre);


  struct data_pipeline* input=data_map(bottom, hd);
  struct data_pipeline* output;

  if(eps<0)
    eps=0.00001;//this is the default value
 
  if(!strcmp(bottom, top))
  {
    output=input;
  }
  else
  {
    if(hd->data_has_init)
    {
      output=data_map(top, hd);
    }
    else
    {
      int nchw[4];
      nchw[0]=input->datas.shape[0];
      nchw[1]=input->datas.shape[1];
      nchw[2]=input->datas.shape[2];
      nchw[3]=input->datas.shape[3];
      output=data_map_init(top, nchw, hd);
      inferx_keep_data_shape(input, output);
    }
  }

  struct model_pipeline* model;

  struct data_arg dg;


  int vw[3], nw, nv[3];
  nw=3;
  nv[0]=1; nv[1]=1; nv[2]=1;
  vw[0]=input->datas.shape[1];
  vw[1]=input->datas.shape[1];
  vw[2]=1;


  model=get_model(iname, nw, nv, vw, dg, hd);



  batch_norm_forward_func(input->datas, output->datas, model->weight[2], model->weight[0], model->weight[1], eps, input->datas.shape[0], input->datas.shape[1], input->datas.shape[2], input->datas.shape[3], hd);

  if(hd->tsw==time_layer)
  {
    struct timeval end_layer;
    gettimeofday(&end_layer, NULL);
    inferx_update_layer_timer(end_layer, iname, hd);
  }

  return;
}


void inferx_scale(int axis,int num_axes, bool bias, char* bottom_pre, char* top_pre, char* iname_pre, char *model_pre, char *data_pre, struct handler *hd)
{
  if(hd->tsw==time_layer)
    gettimeofday(&(hd->start_layer), NULL);

  char iname[1000];
  char bottom[1000];
  char top[1000];
  sprintf(iname, "%s%s", model_pre, iname_pre);
  sprintf(bottom, "%s%s", data_pre, bottom_pre);
  sprintf(top, "%s%s", data_pre, top_pre);


  struct data_pipeline* input=data_map(bottom, hd);
  struct data_pipeline* output;
 
  if(!strcmp(bottom, top))
  {
    output=input;
  }
  else
  {
    if(hd->data_has_init)
    {
      output=data_map(top, hd);
    }
    else
    {
      int nchw[4];
      nchw[0]=input->datas.shape[0];
      nchw[1]=input->datas.shape[1];
      nchw[2]=input->datas.shape[2];
      nchw[3]=input->datas.shape[3];
      output=data_map_init(top, nchw, hd);
      inferx_keep_data_shape(input, output);
    }
  }

  struct model_pipeline* model;

  struct data_arg dg;


  int vw[2], nw, nv[2];
  nw=2;
  nv[0]=1; nv[1]=1;
  vw[0]=input->datas.shape[1];
  vw[1]=input->datas.shape[1];
  model=get_model(iname, nw, nv, vw, dg, hd);


  scale_forward_func(input->datas, output->datas, model->weight[0], model->weight[1], input->datas.shape[0], input->datas.shape[1], input->datas.shape[2], input->datas.shape[3], bias, hd);

  if(hd->tsw==time_layer)
  {
    struct timeval end_layer;
    gettimeofday(&end_layer, NULL);
    inferx_update_layer_timer(end_layer, iname, hd);
  }

  return;
}





void inferx_sigmoid(char* bottom_pre, char* top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd)
{
  if(hd->tsw==time_layer)
    gettimeofday(&(hd->start_layer), NULL);

  char iname[1000];
  char bottom[1000];
  char top[1000];
  sprintf(iname, "%s%s", model_pre, iname_pre);
  sprintf(bottom, "%s%s", data_pre, bottom_pre);
  sprintf(top, "%s%s", data_pre, top_pre);


  struct data_pipeline* input=data_map(bottom, hd);
  struct data_pipeline* output;

  if(!strcmp(bottom, top))
  {
    output=input;
  }
  else
  {
    if(hd->data_has_init)
    {
      output=data_map(top, hd);
    }
    else
    {
      int nchw[4];
      nchw[0]=input->datas.shape[0];
      nchw[1]=input->datas.shape[1];
      nchw[2]=input->datas.shape[2];
      nchw[3]=input->datas.shape[3];
      output=data_map_init(top, nchw, hd);
      inferx_keep_data_shape(input, output);
    }
  }

  sigmoid_forward_func(input->datas, output->datas, input->datas.shape[0]*input->datas.shape[1]*input->datas.shape[2]*input->datas.shape[3], hd);

  if(hd->tsw==time_layer)
  {
    struct timeval end_layer;
    gettimeofday(&end_layer, NULL);
    inferx_update_layer_timer(end_layer, iname, hd);
  }

  return;
}



void inferx_softmax(int axis, char* bottom_pre, char* top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd)
{
  if(hd->tsw==time_layer)
    gettimeofday(&(hd->start_layer), NULL);

  char iname[1000];
  char bottom[1000];
  char top[1000];
  sprintf(iname, "%s%s", model_pre, iname_pre);
  sprintf(bottom, "%s%s", data_pre, bottom_pre);
  sprintf(top, "%s%s", data_pre, top_pre);


  struct data_pipeline* input=data_map(bottom, hd);
  struct data_pipeline* output;

  if(!strcmp(bottom, top))
  {
    output=input;
  }
  else
  {
    if(hd->data_has_init)
    {
      output=data_map(top, hd);
    }
    else
    {
      int nchw[4];
      nchw[0]=input->datas.shape[0];
      nchw[1]=input->datas.shape[1];
      nchw[2]=input->datas.shape[2];
      nchw[3]=input->datas.shape[3];
      output=data_map_init(top, nchw, hd);
      inferx_keep_data_shape(input, output);
    }
  }

  softmax_forward_func(input->datas, output->datas, input->datas.shape[0], input->datas.shape[1]*input->datas.shape[2]*input->datas.shape[3], hd);

  if(hd->tsw==time_layer)
  {
    struct timeval end_layer;
    gettimeofday(&end_layer, NULL);
    inferx_update_layer_timer(end_layer, iname, hd);
  }

  return;
}



void inferx_log_softmax(char* bottom_pre, char* top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd)
{
  if(hd->tsw==time_layer)
    gettimeofday(&(hd->start_layer), NULL);

  char iname[1000];
  char bottom[1000];
  char top[1000];
  sprintf(iname, "%s%s", model_pre, iname_pre);
  sprintf(bottom, "%s%s", data_pre, bottom_pre);
  sprintf(top, "%s%s", data_pre, top_pre);


  struct data_pipeline* input=data_map(bottom, hd);
  struct data_pipeline* output;

  if(!strcmp(bottom, top))
  {
    output=input;
  }
  else
  { 
    if(hd->data_has_init)
    {
      output=data_map(top, hd);
    }
    else
    {
      int nchw[4];
      nchw[0]=input->datas.shape[0];
      nchw[1]=input->datas.shape[1];
      nchw[2]=input->datas.shape[2];
      nchw[3]=input->datas.shape[3];
      output=data_map_init(top, nchw, hd);
      inferx_keep_data_shape(input, output);
    }
  }

  log_softmax_forward_func(input->datas, output->datas, input->datas.shape[0], input->datas.shape[1]*input->datas.shape[2]*input->datas.shape[3], hd);

  if(hd->tsw==time_layer)
  {
    struct timeval end_layer;
    gettimeofday(&end_layer, NULL);
    inferx_update_layer_timer(end_layer, iname, hd);
  }

  return;
}




void inferx_input(int *nchw_l, void *pdata, char* top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd)
{
  if(hd->tsw==time_layer)
    gettimeofday(&(hd->start_layer), NULL);

  if(hd->tsw==time_forward)
    gettimeofday(&(hd->start_forward), NULL);

  char iname[1000];
  char top[1000];
  sprintf(iname, "%s%s", model_pre, iname_pre);
  sprintf(top, "%s%s", data_pre, top_pre);

printf("ooo %d\n", hd->data_has_init);

  if(!(hd->data_has_init))
  {
    struct data_pipeline* input = data_map_init(top, nchw_l, hd);
printf("%d\n", input->datas.ndim);
printf("%d %d %d %d\n", nchw_l[0], nchw_l[1], nchw_l[2], nchw_l[3]);

    input_rand_init(input, hd);
  }
  else
  {
    struct data_pipeline* input = data_map(top, hd);
    input->datas.shape[0]=nchw_l[0];
    input->datas.shape[1]=nchw_l[1];
    input->datas.shape[2]=nchw_l[2];
    input->datas.shape[3]=nchw_l[3];

    if(hd->is_update_input)
    {
      inferx_update_input_data(top, hd);
    }

    if(pdata)
    {
      if(input->datas.ctx.device_type==kCPU)
      {
        input->datas.data=pdata;
      }
      else if(input->datas.ctx.device_type==kGPU)
      {
#ifdef CUDNN
#else
        printf("not define the macro CUDNN, but you are using the library, please define it in the Makefile\n");
        exit(1);
#endif
      }
      else if(input->datas.ctx.device_type==kOpenCL)
      {
#ifdef PERFDNN_OCL
          perf_dnn_mat_t mat_cpu = perf_dnn_init_mat_with_data((void*)pdata, input->datas.shape[2], input->datas.shape[3], input->datas.shape[3], input->datas.shape[1], input->datas.shape[0], PERFDNN_32F);
          if(input->datas.mat_ocl==NULL)
          {
            input->datas.mat_ocl = perf_dnn_init_ocl_mat(hd->perf_ocl_context, input->datas.shape[2], input->datas.shape[3], input->datas.shape[3], input->datas.shape[1], input->datas.shape[0], PERFDNN_32F,0,NULL);
          }
          perf_dnn_upload_data_to_device(hd->perf_ocl_context, mat_cpu, input->datas.mat_ocl, 1, 0);
#else
        printf("not define the macro PERFDNN_OCL, but you are using the library, please define it in the Makefile\n");
        exit(1);
#endif
      }
      else
      {
        printf("can not support the device type %d in input\n", input->datas.ctx.device_type);
      }
    }
  }

  if(hd->tsw==time_layer)
  {
    struct timeval end_layer;
    gettimeofday(&end_layer, NULL);
    inferx_update_layer_timer(end_layer, iname, hd);
  }

  return;
}



void inferx_reshape(int n, int c, int h, int w, char *bottom_pre, char* top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd)
{
  if(hd->tsw==time_layer)
    gettimeofday(&(hd->start_layer), NULL);


  char iname[1000];
  char bottom[1000];
  char top[1000];
  sprintf(iname, "%s%s", model_pre, iname_pre);
  sprintf(bottom, "%s%s", data_pre, bottom_pre);
  sprintf(top, "%s%s", data_pre, top_pre);

  struct data_pipeline* input=data_map(bottom, hd);


  int nchw[4];
  nchw[0]=n;
  nchw[1]=c;
  nchw[2]=h;
  nchw[3]=w;
  int neg_index=-1;
  int i;
  for(i=0; i<4; i++)
  {
    if(nchw[i]==-1)
    {
      neg_index=i;
      nchw[i]=1;
      continue;
    }
    if(nchw[i]==0)
    {
      nchw[i]=input->datas.shape[i];
    }
  }
  
  if(neg_index>=0)
  {
    int len=input->datas.shape[0]*input->datas.shape[1]*input->datas.shape[2]*input->datas.shape[3];
    int lendiv=1;
    lendiv=nchw[0]*nchw[1]*nchw[2]*nchw[3];
    nchw[lendiv]=len/lendiv;
  }

  struct data_pipeline* output;

  if(!(hd->data_has_init))
  {
    output = data_map_init(top, nchw, hd);

    input_rand_init(output, hd);
  }
  else
  {
    output = data_map(top, hd);
  }
  output->datas.data = input->datas.data;

  if(input->datas.ctx.device_type==kGPU)
  {
#ifdef CUDNN
#else
    printf("not define the macro CUDNN, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else if(input->datas.ctx.device_type==kOpenCL)
  {
#ifdef PERFDNN_OCL
    //perf_dnn_reshape_ocl_mat(input->datas.mat_ocl, output->datas.mat_ocl, nchw[2], nchw[3], nchw[1]);
#else
    printf("not define the macro PERFDNN_OCL, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else
  {
    printf("can not support the device type %d in Reshape\n", input->datas.ctx.device_type);
  }


  if(hd->tsw==time_layer)
  {
    struct timeval end_layer;
    gettimeofday(&end_layer, NULL);
    inferx_update_layer_timer(end_layer, iname, hd);
  }


  return;
}



void inferx_reorg(int n, int c, int h, int w, char *bottom_pre, char* top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd)
{
  if(hd->tsw==time_layer)
    gettimeofday(&(hd->start_layer), NULL);


  char iname[1000];
  char bottom[1000];
  char top[1000];
  sprintf(iname, "%s%s", model_pre, iname_pre);
  sprintf(bottom, "%s%s", data_pre, bottom_pre);
  sprintf(top, "%s%s", data_pre, top_pre);

  struct data_pipeline* input=data_map(bottom, hd);


  int nchw[4];
  nchw[0]=n;
  nchw[1]=c;
  nchw[2]=h;
  nchw[3]=w;
  int neg_index=-1;
  int i;
  for(i=0; i<4; i++)
  {
    if(nchw[i]==-1)
    {
      neg_index=i;
      nchw[i]=1;
      continue;
    }
    if(nchw[i]==0)
    {
      nchw[i]=input->datas.shape[i];
    }
  }
  
  if(neg_index>=0)
  {
    int len=input->datas.shape[0]*input->datas.shape[1]*input->datas.shape[2]*input->datas.shape[3];
    int lendiv=1;
    lendiv=nchw[0]*nchw[1]*nchw[2]*nchw[3];
    nchw[lendiv]=len/lendiv;
  }

  struct data_pipeline* output;

  if(!(hd->data_has_init))
  {
    output = data_map_init(top, nchw, hd);

    input_rand_init(output, hd);
  }
  else
  {
    output = data_map(top, hd);
  }
  int stride=2;


  reorg_forward_func(input->datas, output->datas, input->datas.shape[0], input->datas.shape[1], input->datas.shape[2], input->datas.shape[3], stride, 0, hd);


  if(hd->tsw==time_layer)
  {
    struct timeval end_layer;
    gettimeofday(&end_layer, NULL);
    inferx_update_layer_timer(end_layer, iname, hd);
  }

  return;
}





void inferx_print_data(char *bottom_pre, char *data_pre, struct handler *hd)
{
  char bottom[1000];
  sprintf(bottom, "%s%s", data_pre, bottom_pre);

  struct data_pipeline* input=data_map(bottom, hd);
  int len=input->datas.shape[0]*input->datas.shape[1]*input->datas.shape[2]*input->datas.shape[3];


  if(input->datas.ctx.device_type==kCPU)
  {
    if(input->datas.dtype.code==2&&input->datas.dtype.lanes==1)
    {
      int i;
      float *p=(float*)input->datas.data;
      int pl=(10<len?10:len);
      for(i=0; i<len; i++)
      {
			//if ( i % (112 *112 ) == 0)
		  if (i < 5)
			printf("%d  :  %f\n",i, p[i]);
      }
      printf("----------------\n");
    }
  }
  else if(input->datas.ctx.device_type==kGPU)
  {
#ifdef CUDNN
#else
    printf("not define the macro CUDNN, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else if(input->datas.ctx.device_type==kOpenCL)
  {
#ifdef PERFDNN_OCL
    perf_dnn_mat_t mat_cpu = perf_dnn_init_mat_with_data(input->datas.data, input->datas.shape[2], input->datas.shape[3], input->datas.shape[3], input->datas.shape[1], input->datas.shape[0], PERFDNN_32F);
    perf_dnn_download_data_from_device(hd->perf_ocl_context, input->datas.mat_ocl, mat_cpu, 1, 0); // the first 0 means asyn, 1 syn
    int i;
    float *p=(float*)(mat_cpu->data);
    int pl=(10<len?10:len);
    for(i=0; i<len; i++)
    {
      printf("%f\n", p[i]);
    }
    printf("----------------\n");
#else
    printf("not define the macro PERFDNN_OCL, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else
  {
    printf("can not support the device type %d in printData\n", input->datas.ctx.device_type);
  }

  return;
}


void inferx_sort_data(char *bottom_pre, char *data_pre, struct handler *hd)
{
  char bottom[1000];
  sprintf(bottom, "%s%s", data_pre, bottom_pre);

  struct data_pipeline* input=data_map(bottom, hd);
  int len=input->datas.shape[0]*input->datas.shape[1]*input->datas.shape[2]*input->datas.shape[3];

  if(input->datas.dtype.code==2&&input->datas.dtype.lanes==1)
  {
    quick_sort_descend((float*)(input->datas.data), 0, len-1);
  }

  return;
}

void inferx_save_data(char *path, char *bottom_pre, char *data_pre, struct handler *hd)
{
  char bottom[1000];
  sprintf(bottom, "%s%s", data_pre, bottom_pre);

  struct data_pipeline* input=data_map(bottom, hd);
  FILE *fp;
  int len=input->datas.shape[0]*input->datas.shape[1]*input->datas.shape[2]*input->datas.shape[3];

  char name[50];
  sprintf(name, "%s/%s.txt", path, bottom);

  if((fp=fopen(name, "wt"))==NULL)
  {
    printf("can not open the file %s\n", name);
    exit(1);
  }


  if(input->datas.ctx.device_type==kCPU)
  {
    if(input->datas.dtype.code==2&&input->datas.dtype.lanes==1)
    {
      int i;
      float *p=(float*)input->datas.data;
      for(i=0; i<len; i++)
      {
        fprintf(fp, "%f\n", p[i]);
      }
    }
  }
  else if(input->datas.ctx.device_type==kGPU)
  {
#ifdef CUDNN
#else
    printf("not define the macro CUDNN, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else if(input->datas.ctx.device_type==kOpenCL)
  {
#ifdef PERFDNN_OCL
    perf_dnn_mat_t mat_cpu = perf_dnn_init_mat_with_data(input->datas.data, input->datas.shape[2], input->datas.shape[3], input->datas.shape[3], input->datas.shape[1], input->datas.shape[0], PERFDNN_32F);
    perf_dnn_download_data_from_device(hd->perf_ocl_context, input->datas.mat_ocl, mat_cpu, 1, 0);
    int i;
    float *p=(float*)(mat_cpu->data);
    for(i=0; i<len; i++)
    {
      fprintf(fp, "%f\n", p[i]);
    }
#else
    printf("not define the macro PERFDNN_OCL, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else
  {
    printf("can not support the device type %d in saveData\n", input->datas.ctx.device_type);
  }

  fclose(fp);

  return;
}


void* inferx_get_data(char *bottom_pre, char *data_pre, struct handler *hd)
{
  char bottom[1000];
  sprintf(bottom, "%s%s", data_pre, bottom_pre);

  struct data_pipeline* input=data_map(bottom, hd);
  
  void *p=NULL;

  if(input->datas.ctx.device_type==kCPU)
  {
    p=input->datas.data;
  }
  else if(input->datas.ctx.device_type==kGPU)
  {
#ifdef CUDNN
#else
    printf("not define the macro CUDNN, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else if(input->datas.ctx.device_type==kOpenCL)
  {
#ifdef PERFDNN_OCL
    perf_dnn_mat_t mat_cpu = perf_dnn_init_mat_with_data(input->datas.data, input->datas.shape[2], input->datas.shape[3], input->datas.shape[3], input->datas.shape[1], input->datas.shape[0], PERFDNN_32F);
    perf_dnn_download_data_from_device(hd->perf_ocl_context, input->datas.mat_ocl, mat_cpu, 1, 0);
    p=mat_cpu->data;
#else
    printf("not define the macro PERFDNN_OCL, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else
  {
    printf("can not support the device type %d in saveData\n", input->datas.ctx.device_type);
  }

  return p;
}



void* inferx_get_model(char *bottom_pre, char *data_pre, struct handler *hd)
{
  char bottom[1000];
  sprintf(bottom, "%s%s", data_pre, bottom_pre);

  printf("%s\n", bottom);

  struct model_pipeline* input=weight_bias_map(bottom, hd);

  void *p;

  if(input->weight[0].ctx.device_type==kCPU)
  {
    p=input->weight[0].data;

  }
  else if(input->weight[0].ctx.device_type==kGPU)
  {
#ifdef CUDNN
#else
    printf("not define the macro CUDNN, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else if(input->weight[0].ctx.device_type==kOpenCL)
  {
#ifdef PERFDNN_OCL
    perf_dnn_mat_t mat_cpu = perf_dnn_init_mat_with_data(input->weight[0].data, input->weight[0].shape[2], input->weight[0].shape[3], input->weight[0].shape[3], input->weight[0].shape[1], input->weight[0].shape[0], PERFDNN_32F);
    perf_dnn_download_data_from_device(hd->perf_ocl_context, input->weight[0].mat_ocl, mat_cpu, 1, 0);
    p=mat_cpu->data;
#else
    printf("not define the macro PERFDNN_OCL, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else
  {
    printf("can not support the device type %d in saveData\n", input->weight[0].ctx.device_type);
  }

  return p;
}


void inferx_concat(int num_output,int axis,int concat_dim,int bottom_num,char **bottoms_pre,char *top_pre,char *iname_pre, char *model_pre, char *data_pre, struct handler *hd)
{
  if(hd->tsw==time_layer)
    gettimeofday(&(hd->start_layer), NULL);

  char iname[1000];
  char **bottoms;
  char top[1000];
  sprintf(iname, "%s%s", model_pre, iname_pre);
  bottoms=(char**)malloc(sizeof(char*)*bottom_num);
  int i;
  for(i=0; i<bottom_num; i++)
  {
    bottoms[i]=(char*)malloc(sizeof(char)*1000);
    sprintf(bottoms[i], "%s%s", data_pre, bottoms_pre[i]);
  }
  sprintf(top, "%s%s", data_pre, top_pre);


  DLTensor *input=(DLTensor*)malloc(sizeof(DLTensor)*bottom_num);
  for(i=0; i<bottom_num; i++)
  {
    struct data_pipeline* p=data_map(bottoms[i], hd);
    input[i]=p->datas;
  }
  struct data_pipeline* output;
  if(hd->data_has_init)
  {
    output=data_map(top, hd);
  }
  else
  {
    int n=0, c=0, h=0, w=0;
    n=input[0].shape[0];
    c=input[0].shape[1];
    h=input[0].shape[2];
    w=input[0].shape[3];
    switch(axis)
    {
      case 0: for(i=1; i<bottom_num; i++){n+=input[i].shape[0];}; break;
      case 1: for(i=1; i<bottom_num; i++){c+=input[i].shape[1];}; break;
      case 2: for(i=1; i<bottom_num; i++){h+=input[i].shape[2];}; break;
      case 3: for(i=1; i<bottom_num; i++){w+=input[i].shape[3];}; break;
      default: printf("unsupported concact methods\n");
    }
    int nchw[4];
    nchw[0]=n;
    nchw[1]=c;
    nchw[2]=h;
    nchw[3]=w;
    output=data_map_init(top, nchw, hd);
  }
  tensor_concat_func(axis, bottom_num, input, output->datas, hd);

  for(i=0; i<bottom_num; i++)
  {
    free((void*)(bottoms[i]));
  }
  free((void*)bottoms);

  if(hd->tsw==time_layer)
  {
    struct timeval end_layer;
    gettimeofday(&end_layer, NULL);
    inferx_update_layer_timer(end_layer, iname, hd);
  }

  return;
}




void inferx_slice(int axis, char *bottom_pre, char *top1_pre, char *top2_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd)
{
  if(hd->tsw==time_layer)
    gettimeofday(&(hd->start_layer), NULL);

  char iname[1000];
  char bottom[1000];
  char top1[1000], top2[1000];
  sprintf(iname, "%s%s", model_pre, iname_pre);
  sprintf(bottom, "%s%s", data_pre, bottom_pre);
  sprintf(top1, "%s%s", data_pre, top1_pre);
  sprintf(top2, "%s%s", data_pre, top2_pre);


  if(hd->tsw==time_layer)
  {
    struct timeval end_layer;
    gettimeofday(&end_layer, NULL);
    inferx_update_layer_timer(end_layer, iname, hd);
  }

  return;
}

void inferx_eltwise(int coeffs_num, float * coeffs, enum OPERATION op, bool stabel_prod_grad, int bottom_num, char ** bottoms_pre, char * top_pre, char * iname_pre, char * model_pre, char * data_pre, struct handler * hd)
{
	inferx_elem_wise_operate(coeffs_num, coeffs, op, stabel_prod_grad, bottom_num, bottoms_pre, top_pre, iname_pre, model_pre, data_pre, hd);

}


void inferx_lrn(int local_size, float alpha, float beta, char *bottom_pre, char *top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd)
{
  if(hd->tsw==time_layer)
    gettimeofday(&(hd->start_layer), NULL);

  char iname[1000];
  char bottom[1000];
  char top[1000];
  sprintf(iname, "%s%s", model_pre, iname_pre);
  sprintf(bottom, "%s%s", data_pre, bottom_pre);
  sprintf(top, "%s%s", data_pre, top_pre);


  enum LRN_WAY lrn=across_channels;

  struct data_pipeline* input=data_map(bottom, hd);
  struct data_pipeline* output;
  if(hd->data_has_init)
  {
    output=data_map(top, hd);
  }
  else
  {
    int nchw[4];
    nchw[0]=input->datas.shape[0];
    nchw[1]=input->datas.shape[1];
    nchw[2]=input->datas.shape[2];
    nchw[3]=input->datas.shape[3];
    output=data_map_init(top, nchw, hd);
    inferx_keep_data_shape(input, output);
  }

  inferx_zero_data(output);

  lrn_forward_func(input->datas, output->datas, lrn, local_size, alpha, beta, input->datas.shape[1], input->datas.shape[2], input->datas.shape[3], hd);

  if(hd->tsw==time_layer)
  {
    struct timeval end_layer;
    gettimeofday(&end_layer, NULL);
    inferx_update_layer_timer(end_layer, iname, hd);
  }

  return;
}



void inferx_elem_wise_operate(int coeffs_num, float* coeffs,enum OPERATION op,bool stabel_prod_grad,int bottom_num,char **bottoms_pre,char *top_pre, char *iname_pre, char *model_pre, char *data_pre, struct handler *hd)
{
  if(hd->tsw==time_layer)
    gettimeofday(&(hd->start_layer), NULL);

  char iname[1000];
  char **bottoms;
  char top[1000];
  sprintf(iname, "%s%s", model_pre, iname_pre);
  bottoms=(char**)malloc(sizeof(char*)*bottom_num);
  int i;
  for(i=0; i<bottom_num; i++)
  {
    bottoms[i]=(char*)malloc(sizeof(char)*1000);
    sprintf(bottoms[i], "%s%s", data_pre, bottoms_pre[i]);
  }
  sprintf(top, "%s%s", data_pre, top_pre);


  DLTensor *input=(DLTensor*)malloc(sizeof(DLTensor)*bottom_num);
  struct data_pipeline* p=NULL;
  for(i=0; i<bottom_num; i++)
  {
    p=data_map(bottoms[i], hd);
    input[i]=p->datas;
  }
  struct data_pipeline* output;


  if(hd->data_has_init)
  {
    output=data_map(top, hd);
  }
  else
  {
    int nchw[4];
    nchw[0]=p->datas.shape[0];
    nchw[1]=p->datas.shape[1];
    nchw[2]=p->datas.shape[2];
    nchw[3]=p->datas.shape[3];
    output=data_map_init(top, nchw, hd);
    inferx_keep_data_shape(p, output);
  }

  elem_wise_operate_func(bottom_num, input, output->datas, (int)(p->datas.shape[0]*p->datas.shape[1]*p->datas.shape[2]*p->datas.shape[3]), op, hd);

  for(i=0; i<bottom_num; i++)
  {
    free((void*)(bottoms[i]));
  }
  free((void*)bottoms);


  free((void*)input);

  if(hd->tsw==time_layer)
  {
    struct timeval end_layer;
    gettimeofday(&end_layer, NULL);
    inferx_update_layer_timer(end_layer, iname, hd);
  }

  return;
}



void inferx_crop(int axis,int offset,char* bottom_pre, char* bottom_mode_pre,char *top_pre,char *iname_pre, char *model_pre, char *data_pre, struct handler *hd)
{
  if(hd->tsw==time_layer)
    gettimeofday(&(hd->start_layer), NULL);

  char iname[1000];
  char bottom[1000];
  char bottom_mode[1000];
  char top[1000];
  sprintf(iname, "%s%s", model_pre, iname_pre);
  sprintf(bottom, "%s%s", data_pre, bottom_pre);
  sprintf(bottom_mode, "%s%s", data_pre, bottom_mode_pre);
  sprintf(top, "%s%s", data_pre, top_pre);


  struct data_pipeline* mode=data_map(bottom_mode, hd);
  int top_n=mode->datas.shape[0];
  int top_c=mode->datas.shape[1];
  int top_h=mode->datas.shape[2];
  int top_w=mode->datas.shape[3];

  struct data_pipeline* input=data_map(bottom, hd);
  struct data_pipeline* output;
  int offset_n=0, offset_c=0, offset_h=0, offset_w=0;

  switch(axis)
  {
    case 0: offset_n=offset;
    case 1: offset_c=offset;
    case 2: offset_h=offset;
    case 3: offset_w=offset; break;
    default: printf("unsupported axis\n");
  }

  if(!strcmp(bottom, top))
  {
    output=input;
  }
  else
  {
    if(hd->data_has_init)
    {
      output=data_map(top, hd);
    }
    else
    {
      int nchw[4];
      nchw[0]=top_n;
      nchw[1]=top_c;
      nchw[2]=top_h;
      nchw[3]=top_w;
      output=data_map_init(top, nchw, hd);
    }
  }

  crop_forward_func(input->datas, output->datas, axis, input->datas.shape[0], input->datas.shape[1], input->datas.shape[2], input->datas.shape[3], output->datas.shape[0], output->datas.shape[1], output->datas.shape[2], output->datas.shape[3], offset_n, offset_c, offset_h, offset_w, hd);

  if(hd->tsw==time_layer)
  {
    struct timeval end_layer;
    gettimeofday(&end_layer, NULL);
    inferx_update_layer_timer(end_layer, iname, hd);
  }

  return;
}


void inferx_finalize(char* modelname, struct handler *hd)
{
  struct timeval end_forward;
  double ftime;

  hd->data_has_init=true;


#ifdef PERFDNN_OCL
  clFinish(hd->perf_ocl_context->command_queue);
#endif

  
  if(hd->tsw==time_forward)
  {
    gettimeofday(&end_forward, NULL);
    ftime = (end_forward.tv_sec-hd->start_forward.tv_sec) * 1000 + (double)(end_forward.tv_usec-hd->start_forward.tv_usec)/1000;
    printf("model %s forward run time %f\n", modelname, ftime);
  }
  if(hd->tsw==time_layer)
  {
    hd->time_tail->tp=NULL;
    hd->time_tail=hd->time_head;
    hd->time_layer_cnt++;
  }

  return;
}



void inferx_net_preprocess(char *data, char *model, int nchw[4], struct handler *hd)
{
  static int cnt;
  if(cnt==123456)
  {
    inferx_set_init_var(&(hd->weight_has_load), &(hd->data_has_init), model, data, hd);
    for (int i = 0; i<4; i++)
    {
        printf("nchw[%d]:%d\n", i, nchw[i]);
    }
    return;
  }
  inferx_parse_str(data, nchw);
  inferx_set_init_var(&(hd->weight_has_load), &(hd->data_has_init), model, data, hd);
  inferx_var_add_init(model, hd);
  inferx_var_add_init(data, hd);
  strcpy(data+strlen(data), model);


  cnt=123456;

  return;
}
