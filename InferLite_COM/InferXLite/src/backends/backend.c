#include "backend.h"
#include "backend_impl.h"
#include "inferxlite_common.h"
#include "math.h"
#ifdef PERFDNN
#include "perfdnn.h"
#endif


#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON



void convolution_forward_func(struct CONV_ARG *conv_arg_pt, const DLTensor input_t, DLTensor *filter_t1, const DLTensor filter_t2, DLTensor output_t, bool *bias, int activation_type, struct handler *hd)
{
  if(input_t.ctx.device_type==kCPU)
  {
    if(hd->cw==conv_nature)
    {
      if(conv_arg_pt->dila_h!=1||conv_arg_pt->dila_w!=1)
      {
        printf("can not support the dilation argument for direct convolution\n");
        exit(1);
      }

      int ilen=0, olen=0, wlen=0;
      if(conv_arg_pt->group>1)
      {
        ilen=input_t.shape[0]*input_t.shape[1]*input_t.shape[2]*input_t.shape[3]/conv_arg_pt->group;
        olen=output_t.shape[0]*output_t.shape[1]*output_t.shape[2]*output_t.shape[3]/conv_arg_pt->group;
        wlen=conv_arg_pt->input_c*conv_arg_pt->output_c*conv_arg_pt->kernel_h*conv_arg_pt->kernel_w/conv_arg_pt->group/conv_arg_pt->group;
      }
    
      for(int g=0; g<conv_arg_pt->group; g++)
      {
        float *input=(float*)input_t.data+g*ilen;
        float *filter=(float*)filter_t1->data+g*wlen;
        float *output=(float*)output_t.data+g*olen;
    
        if(conv_arg_pt->pad_h == 0 && conv_arg_pt->pad_w == 0)
        {
          convolution_forward_naive_no_pad_nchw_impl(conv_arg_pt, input, filter, output);
        }
        else
        {
          convolution_forward_naive_pad_nchw_impl(conv_arg_pt, input, filter, output);
        }
      }
    }
    else if(hd->cw==conv_gemm)
    {
      float *input=(float*)input_t.data;
      float *filter=(float*)filter_t1->data;
      float *output=(float*)output_t.data;
  
      convolution_forward_gemm_nchw_impl(conv_arg_pt, input, filter, output);
    }
    else if(hd->cw==conv_perfdnn)
    {
      float *input=(float*)input_t.data;
      float *filter=(float*)filter_t1->data;
      float *bias=(float*)filter_t2.data;
      float *output=(float*)output_t.data;

      convolution_forward_perfdnn_nchw_impl(conv_arg_pt, input, filter, bias, output, bias, activation_type);
    }
    else
    {
      printf("unsupported convolution ways\n");
    }
  }
  else if(input_t.ctx.device_type==kGPU)
  {
#ifdef CUDNN
    float alpha=1.0;
    float beta=0.0;
    struct context_data *desc_i = (struct context_data*)input_t.desc;
    struct context_model *desc_w = (struct context_model*)filter_t1->desc;
    struct context_data *desc_o = (struct context_data*)output_t.desc;

    if(!hd->weight_has_load)
    {
      cudnnGetConvolutionForwardWorkspaceSize(hd->hCudNN, desc_i->pDataDesc, desc_w->pFilterDesc, desc_w->pConvDesc, desc_o->pDataDesc, CUDNN_CONVOLUTION_FWD_ALGO_GEMM, &(desc_w->sizeInBytes));
      cudaMalloc(&(desc_w->workSpace),desc_w->sizeInBytes);
    }

    cudnnStatus_t status;
    status = cudnnConvolutionForward(hd->hCudNN, &alpha, desc_i->pDataDesc, input_t.data, desc_w->pFilterDesc, filter_t1->data, desc_w->pConvDesc, CUDNN_CONVOLUTION_FWD_ALGO_GEMM, desc_w->workSpace, desc_w->sizeInBytes, &beta, desc_o->pDataDesc, output_t.data);
    if (status != CUDNN_STATUS_SUCCESS)
      printf("cudnn error %d\n", status);
#else
    printf("not define the macro CUDNN, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else if(input_t.ctx.device_type==kOpenCL)
  {
#ifdef PERFDNN_OCL
    if(filter_t1->tmp_ocl==NULL)
    {
       int rows=conv_arg_pt->kernel_h*conv_arg_pt->kernel_w*input_t.shape[1];
       int cols=output_t.shape[2]*output_t.shape[3];
       filter_t1->tmp_ocl = perf_dnn_init_ocl_mat(hd->perf_ocl_context, rows, cols, cols, 1, 1, PERFDNN_32F,0,NULL);
    }
    perf_dnn_conv(hd->perf_ocl_context, input_t.mat_ocl, filter_t1->mat_ocl, filter_t1->tmp_ocl, output_t.mat_ocl, conv_arg_pt->kernel_h, conv_arg_pt->str_h, conv_arg_pt->pad_h);
#else
    printf("not define the macro PERFDNN_OCL, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else
  {
    printf("can not support the device type %d in convolutionForwardFunc\n", input_t.ctx.device_type);
  }

  return;
}


void convolution_backward_func(struct CONV_ARG *conv_arg_pt, const DLTensor output_t, const DLTensor filter_t, DLTensor input_t, struct handler *hd)
{
  if(input_t.ctx.device_type==kCPU)
  {
    if(hd->cw==conv_nature)
    {
      float *output=(float*)output_t.data;
      float *filter=(float*)filter_t.data;
      float *input=(float*)input_t.data;

      if(conv_arg_pt->pad_h == 0 && conv_arg_pt->pad_w == 0)
      {
        convolution_backward_naive_no_pad_nchw_impl(conv_arg_pt, input, filter, output);
      }
      else
      {
        convolution_backward_naive_pad_nchw_impl(conv_arg_pt, input, filter, output);
      }
    }
    else if(hd->cw==conv_gemm)
    {
      float *input=(float*)input_t.data;
      float *filter=(float*)filter_t.data;
      float *output=(float*)output_t.data;

      convolution_backward_gemm_nchw_impl(conv_arg_pt, output, filter, input);
    }
    else
    {
      printf("unsupported deconvolution ways %d\n", hd->cw);
    }
  }
  else if(input_t.ctx.device_type==kGPU)
  {
#ifdef CUDNN
    const float alpha=1.0;
    const float beta=0.0;
    struct context_data *desc_i = (struct context_data*)input_t.desc;
    struct context_model *desc_w = (struct context_model*)filter_t.desc;
    struct context_data *desc_o = (struct context_data*)output_t.desc;

    if(!hd->weight_has_load)
    {
      cudnnGetConvolutionForwardWorkspaceSize(hd->hCudNN, desc_i->pDataDesc, desc_w->pFilterDesc, desc_w->pConvDesc, desc_o->pDataDesc, CUDNN_CONVOLUTION_FWD_ALGO_GEMM, &(desc_w->sizeInBytes));
      cudaMalloc(&(desc_w->workSpace),desc_w->sizeInBytes);
    }

    cudnnStatus_t status;
    status = cudnnConvolutionBackwardData(hd->hCudNN, &alpha, desc_w->pFilterDesc, filter_t.data, desc_o->pDataDesc, output_t.data, desc_w->pConvDesc, CUDNN_CONVOLUTION_FWD_ALGO_GEMM, desc_w->workSpace, desc_w->sizeInBytes, &beta, desc_i->pDataDesc, input_t.data);

    if (status != CUDNN_STATUS_SUCCESS)
      printf("cudnn error %d\n", status);
#else
    printf("not define the macro CUDNN, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else
  {
    printf("can not support the device type %d in convolutionBackwardFunc\n", input_t.ctx.device_type);
  }

  return;
}



void pooling_forward_func(struct CONV_ARG *conv_arg_pt, const DLTensor input_t, DLTensor output_t, enum OPERATION pool, struct handler *hd)
{
  if(input_t.ctx.device_type==kCPU)
  {
    if(pool==MAX)
    {
      float *input=(float*)input_t.data;
      float *output=(float*)output_t.data;

#if __ARM_NEON
      max_pooling_forward_nc3x3s2_impl(conv_arg_pt, input, output);
#else
      max_pooling_forward_nchw_impl(conv_arg_pt, input, output);
#endif
    }
    else if(pool==AVE)
    {
      float *input=(float*)input_t.data;
      float *output=(float*)output_t.data;

      ave_pooling_forward_nchw_impl(conv_arg_pt, input, output);
    }
    else
    {
      printf("can not support the way of pool\n");
      exit(1);
    }
  }
  else if(input_t.ctx.device_type==kGPU)
  {
#ifdef CUDNN
#else
    printf("not define the macro CUDNN, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else if(input_t.ctx.device_type==kOpenCL)
  {
#ifdef PERFDNN_OCL
    perf_dnn_pooling(hd->perf_ocl_context, input_t.mat_ocl, output_t.mat_ocl, conv_arg_pt->str_h);
#else
    printf("not define the macro PERFDNN_OCL, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else
  {
    printf("can not support the device type %d in poolingForwardFunc_NCHW\n", input_t.ctx.device_type);
  }

  return;
}


void pooling_forward_func_yolo(struct CONV_ARG *conv_arg_pt, const DLTensor input_t, DLTensor output_t, enum OPERATION pool, struct handler *hd)
{
  if(input_t.ctx.device_type==kCPU)
  {
    float *input=(float*)input_t.data;
    float *output=(float*)output_t.data;
    if(pool==MAX)
    {
      max_pooling_forward_nchw_yolo_impl(conv_arg_pt, input, output);
    }
    else if(pool==AVE)
    {
      ave_pooling_forward_nchw_impl(conv_arg_pt, input, output);
    }
    else
    {
      printf("can not support the way of pool\n");
      exit(1);
    }
  }
  else if(input_t.ctx.device_type==kGPU)
  {
#ifdef CUDNN
#else
    printf("not define the macro CUDNN, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else if(input_t.ctx.device_type==kOpenCL)
  {
#ifdef PERFDNN_OCL
    perf_dnn_pooling(hd->perf_ocl_context, input_t.mat_ocl, output_t.mat_ocl, conv_arg_pt->str_h);
#else
    printf("not define the macro PERFDNN_OCL, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else
  {
    printf("can not support the device type %d in poolingForwardFunc_NCHW\n", input_t.ctx.device_type);
  }

  return;
}


void lrn_forward_func(const DLTensor input_t, DLTensor output_t, enum LRN_WAY lrn, int local_size, float alpha, float beta, int channels, int height, int width, struct handler *hd)
{
  float *input=(float*)input_t.data;
  float *output=(float*)output_t.data;

  if(lrn==across_channels)
  {
    lrn_across_forward_impl(input, output, local_size, alpha, beta, channels, height, width);
  }
  else if(lrn==within_channels)
  {
    lrn_within_forward_impl(input, output, local_size, alpha, beta, channels, height, width);
  }
  else
  {
    printf("unsupported lrn norm ways\n");
  }

  return;
}


void relu_forward_func(const DLTensor input_t, DLTensor output_t, int ne, float slope, struct handler *hd)
{
  if(input_t.ctx.device_type==kCPU)
  {
    float *input=(float*)input_t.data;
    float *output=(float*)output_t.data;

    relu_forward_impl(input, output, ne, slope);
  }
  else if(input_t.ctx.device_type==kGPU)
  {
#ifdef CUDNN
#else
    printf("not define the macro CUDNN, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else if(input_t.ctx.device_type==kOpenCL)
  {
#ifdef PERFDNN_OCL
    perf_dnn_activate(hd->perf_ocl_context, input_t.mat_ocl, output_t.mat_ocl, slope);
#else
    printf("not define the macro PERFDNN_OCL, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else
  {
    printf("can not support the device type %d in rectlinApplyFunc\n", input_t.ctx.device_type);
  }

  return;
}

void power_forward_func(const DLTensor input_t, DLTensor output_t, int nchw, float power, float scale, float shift, struct handler *hd)
{
  float *input=(float*)input_t.data;
  float *output=(float*)output_t.data;

  power_forward_impl(input, output, nchw, power, scale, shift);

  return;
}

void tanh_forward_func(const DLTensor input_t, DLTensor output_t, int nchw, struct handler *hd)
{
  float *input=(float*)input_t.data;
  float *output=(float*)output_t.data;

  tanh_forward_impl(input, output, nchw);

  return;
}



void sigmoid_forward_func(const DLTensor input_t, DLTensor output_t, int nchw, struct handler *hd)
{
  float *input=(float*)input_t.data;
  float *output=(float*)output_t.data;

  sigmoid_forward_impl(input, output, nchw);

  return;
}


void softmax_forward_func(const DLTensor input_t, DLTensor output_t, int n, int chw, struct handler *hd)
{
  float *input=(float*)input_t.data;
  float *output=(float*)output_t.data;

  softmax_forward_impl(input, output, n, chw);

  return;
}



void log_softmax_forward_func(const DLTensor input_t, DLTensor output_t, int n, int chw, struct handler *hd)
{
  float *input=(float*)input_t.data;
  float *output=(float*)output_t.data;

  log_softmax_forward_impl(input, output, n, chw);

  return;
}




void bias_forward_func(const DLTensor input_t, const DLTensor bias_t, DLTensor output_t, int n, int k, int output_hw, struct handler *hd)
{
  float *input=(float*)input_t.data;
  float *bias=(float*)bias_t.data;
  float *output=(float*)output_t.data;

  if(input_t.ctx.device_type==kCPU)
  {
    bias_forward_impl(input, bias, output, k, output_hw);
  }
  else if(input_t.ctx.device_type==kGPU)
  {
#ifdef CUDNN
#else
    printf("not define the macro CUDNN, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else if(input_t.ctx.device_type==kOpenCL)
  {
#ifdef PERFDNN_OCL
    perf_dnn_bias(hd->perf_ocl_context, input_t.mat_ocl, bias_t.mat_ocl, output_t.mat_ocl);
#else
    printf("not define the macro PERFDNN_OCL, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else
  {
    printf("can not support the device type %d in biasForwardFunc\n", input_t.ctx.device_type);
  }

  return;
}



void batch_norm_forward_func(const DLTensor input_t, DLTensor output_t, DLTensor scale_factor_t, const DLTensor bn_scale1_t, const DLTensor bn_scale2_t, float eps, int n, int c, int h, int w, struct handler *hd)
{
  float scale_factor;
  if(input_t.ctx.device_type==kCPU)
  {
    float *input=(float*)input_t.data;
    float *output=(float*)output_t.data;
    float *bn_scale1=(float*)bn_scale1_t.data;
    float *bn_scale2=(float*)bn_scale2_t.data;

    scale_factor=((float*)scale_factor_t.data)[0];

    batch_norm_forward_impl(input, output, scale_factor, bn_scale1, bn_scale2, eps, n, c, h, w);
  }
  else if(input_t.ctx.device_type==kGPU)
  {
#ifdef CUDNN
#else
    printf("not define the macro CUDNN, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else if(input_t.ctx.device_type==kOpenCL)
  {
#ifdef PERFDNN_OCL
    scale_factor=((float*)scale_factor_t.data)[0];
    perf_dnn_batchnorm(hd->perf_ocl_context, output_t.mat_ocl, input_t.mat_ocl, scale_factor, bn_scale2_t.mat_ocl, bn_scale1_t.mat_ocl);
#else
    printf("not define the macro PERFDNN_OCL, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else
  {
    printf("can not support the device type %d in batchNormForwardFunc\n", input_t.ctx.device_type);
  }

  return;
}


void scale_forward_func(const DLTensor input_t, DLTensor output_t, const DLTensor gama_t, const DLTensor beta_t, int n, int c, int h, int w, bool is_bias, struct handler *hd)
{
  if(input_t.ctx.device_type==kCPU)
  {
    float *input=(float*)input_t.data;
    float *output=(float*)output_t.data;
    float *gama=(float*)gama_t.data;
    float *beta=(float*)beta_t.data;

    scale_forward_impl(input, output, gama, beta, n, c, h, w, is_bias);
  }
  else if(input_t.ctx.device_type==kGPU)
  {
#ifdef CUDNN
#else
    printf("not define the macro CUDNN, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else if(input_t.ctx.device_type==kOpenCL)
  {
#ifdef PERFDNN_OCL
    perf_dnn_scale(hd->perf_ocl_context, output_t.mat_ocl, input_t.mat_ocl, gama_t.mat_ocl, beta_t.mat_ocl, (int)is_bias);
#else
    printf("not define the macro PERFDNN_OCL, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else
  {
    printf("can not support the device type %d in scaleForwardFunc\n", input_t.ctx.device_type);
  }

  return;
}


void matrix_multiply_func(const DLTensor left_t, const DLTensor right_t, DLTensor output_t, bool transa, bool transb, float alpha, float beta, int m_left, int n_left, int m_right, int n_right, struct handler *hd)
{
  float *left=(float*)left_t.data;
  float *right=(float*)right_t.data;
  float *output=(float*)output_t.data;

  int dim_left = transa ? n_left : m_left;
  int dim_right = transb ? m_right : n_right;
  int dim_common = transa ? m_left : n_left;

  gemm(left, right, output, transa, transb, alpha, beta, dim_left, dim_right, dim_common);
}




/*
float evaluate_classify_forward_func(const DLTensor output_t, const DLTensor target_t, int n, int len)
{
  float *output=(float*)output_t.data;
  float *target=(float*)target_t.data;

  float correct = 0.0f;
  int i, j;
  for(i = 0; i < n; ++i)
  {
    float max = output[i*n];
    int max_index = 0;
    for(j = 1; j < n; ++j)
    {
      if(max<output[i*n+j])
      {
        max_index = j;
        max=output[i*n+j];
      }
    }
    if(target[i*n+max_index]==(float)(1))
    {
      correct += 1.0f;
    }
  }

  return correct / n;
}


float evaluate_regress_forward_func(const DLTensor output, const DLTensor target, int n, int len)
{
  float *output=(float*)output_t.data;
  float *target=(float*)target_t.data;

  float result = 0.0;
  float correct = 0.0;
  int i, j;
  for(i = 0; i < n; ++i)
  {
    for(j = 0; j < len; ++j)
    {
      correct += fabs(output[i] - target[i]);
    }
    result += correct;
  }
  return result / n;
}




float cross_entropy_binary_forward_func(const DLTensor input, const DLTensor target, int len)
{
  float *input=(float*)input_t.data;
  float *target=(float*)target_t.data;

  float output = 0;
  int i;
  for(i = 0; i < input->size(); ++i)
  {
    output += -safeLog(input[i]) * target[i] - safeLog(1 - input[i])*(1 - target[i]);
  }
  output /= len;
  return output;
}

*/







void elem_wise_operate_func(int num_input, DLTensor* input_t, DLTensor output_t, int len, enum OPERATION op, struct handler *hd)
{
  float *output=(float*)output_t.data;
  float **input=(float**)malloc(sizeof(float*)*num_input);

  int i;
  for(i=0; i<num_input; i++)
  {
    input[i]=(float*)input_t[i].data;
  }

  elem_wise_operate_impl(num_input, input, output, len, op);
  
  return;
}





void crop_forward_func(const DLTensor input_t, DLTensor output_t,  int axis, int input_n, int input_c, int input_h, int input_w, int output_n, int output_c, int output_h, int output_w, int offset_n, int offset_c, int offset_h, int offset_w, struct handler *hd)
{
  float *input=(float*)input_t.data;
  float *output=(float*)output_t.data;

  crop_forward_impl(input, output, axis, input_n, input_c, input_h, input_w, output_n, output_c, output_h, output_w, offset_n, offset_c, offset_h, offset_w);

  return;
}




void tensor_concat_func(int axis, int num_input, DLTensor* input_t, DLTensor output_t, struct handler *hd)
{
  float **input;
  float *output=(float*)output_t.data;

  input = (float **)malloc(sizeof(float*)*num_input);
  int *n, *c, *h, *w;
  int no, co, ho, wo;
  n=(int*)malloc(sizeof(int)*num_input);
  c=(int*)malloc(sizeof(int)*num_input);
  h=(int*)malloc(sizeof(int)*num_input);
  w=(int*)malloc(sizeof(int)*num_input);
  int i;
  for(i=0; i<num_input; i++)
  {
    input[i]=(float*)(input_t[i].data);
    n[i]=input_t[i].shape[0];
    c[i]=input_t[i].shape[1];
    h[i]=input_t[i].shape[2];
    w[i]=input_t[i].shape[3];
  }
  no=output_t.shape[0];
  co=output_t.shape[1];
  ho=output_t.shape[2];
  wo=output_t.shape[3];


  if(output_t.ctx.device_type==kCPU)
  {
    tensor_concat_impl(axis, num_input, input, output, n, c, h, w, no, co, ho, wo);
  }
  else if(output_t.ctx.device_type==kGPU)
  {
#ifdef CUDNN
#else
    printf("not define the macro CUDNN, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else if(output_t.ctx.device_type==kOpenCL)
  {
#ifdef PERFDNN_OCL
    if(num_input==2)
    {
      perf_dnn_route(hd->perf_ocl_context, input_t[0].mat_ocl, input_t[1].mat_ocl, output_t.mat_ocl);
    }
    else
    {
      printf("not support the mulit input concat for opencl\n");
    }
#else
    printf("not define the macro PERFDNN_OCL, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else
  {
    printf("can not support the device type %d in tensorMultiConcat\n", output_t.ctx.device_type);
  }


  free((void*)n);
  free((void*)c);
  free((void*)h);
  free((void*)w);

  return;
}


void reorg_forward_func(DLTensor input_t, DLTensor output_t, int n, int c, int h, int w, int stride, int forward, struct handler *hd)
{
  float *input=input_t.data;
  float *output=output_t.data;

  if(input_t.ctx.device_type==kCPU)
  {
    reorg_forward_impl(input, input_t.shape[0], input_t.shape[1], input_t.shape[2], input_t.shape[3], stride, 0, output);
  }
  else if(input_t.ctx.device_type==kGPU)
  {
#ifdef CUDNN
#else
    printf("not define the macro CUDNN, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else if(input_t.ctx.device_type==kOpenCL)
  {
#ifdef PERFDNN_OCL
    perf_dnn_reorg(hd->perf_ocl_context, input_t.mat_ocl, output_t.mat_ocl, stride);
#else
    printf("not define the macro PERFDNN_OCL, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else
  {
    printf("can not support the device type %d in Reshape\n", input_t.ctx.device_type);
  }

  return;
}



void tensor_slice_func(int axis, DLTensor input, DLTensor output1, DLTensor output2, struct handler *hd)
{
  printf("not implemented\n");
  exit(1);

  return;
}
