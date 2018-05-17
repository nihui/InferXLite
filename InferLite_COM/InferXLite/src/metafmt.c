#include "inferxlite_common.h"


void meta_rand_init(DLTensor *p, struct handler *hd)
{
  const float rand_scale=2.0e-3;
  int j;
  int len=1;
  for(j=0; j<p->ndim; j++)
    len*=p->shape[j];
  
  switch(p->dtype.code)
  {
    case 0:
    {
      printf("unsupported metaformat, code:%d\n", p->dtype.code);
      break;
    }
    case 2:
    {
      if(p->dtype.lanes==1)
      {
        if(p->ctx.device_type==kCPU)
        {
          float *pt=(float*)(p->data);
          for(j=0; j<len; j++)
          {
            pt[j]=(rand())/(float)RAND_MAX*rand_scale;
            //pt[j]=0.0003;
          }
        }
        else if(p->ctx.device_type==kGPU)
        {
#ifdef CUDNN
          float *pt=(float*)malloc(sizeof(float)*len);
          for(j=0; j<len; j++)
          {
            pt[j]=(rand())/(float)RAND_MAX*rand_scale;
            //pt[j]=0.0003;
          }
          cudaMemcpy(p->data, pt, sizeof(float)*len, cudaMemcpyHostToDevice);
          free((void*)pt);
#else
          printf("not define the macro CUDNN, but you are using the library, please define it in the Makefile\n");
          exit(1);
#endif
        }
        else if(p->ctx.device_type==kOpenCL)
        {
#ifdef PERFDNN_OCL
          float *pt=(float*)(p->data);
          for(j=0; j<len; j++)
          {
            pt[j]=(rand())/(float)RAND_MAX*rand_scale;
            //pt[j]=0.0003;
          }
          perf_dnn_mat_t mat_cpu = perf_dnn_init_mat_with_data((void*)pt, p->shape[2], p->shape[3], p->shape[3], p->shape[1], p->shape[0], PERFDNN_32F);
          perf_dnn_upload_data_to_device(hd->perf_ocl_context, mat_cpu, p->mat_ocl, 1, 0);
#else
          printf("not define the macro PERFDNN_OCL, but you are using the library, please define it in the Makefile\n");
          exit(1);
#endif
        }
        else
        {
          printf("can not support the device type %d in meta_rand_init\n", p->ctx.device_type);
        }
      }
      else
      {
        printf("unsupported metaformat, lanes:%d\n", p->dtype.lanes);
      }
      break;
    }
    default: printf("unsupported metaformat, code:%d\n", p->dtype.code);
  }

  return;
}


void meta_float_to_tensor(float *input, DLTensor tnsr, struct handler *hd)
{
  int j;

  int len=1;
  for(j=0; j<tnsr.ndim; j++)
    len*=tnsr.shape[j];

  switch(tnsr.dtype.code)
  {
    case 0:
    {
      printf("unsupported metaformat, code:%d\n", tnsr.dtype.code);
      break;
    }
    case 2:
    {
      if(tnsr.dtype.lanes==1)
      {
        if(tnsr.ctx.device_type==kCPU)
        {
          float *p=(float*)(tnsr.data);
          memcpy(p, input, sizeof(float)*len);
        }
        else if(tnsr.ctx.device_type==kGPU)
        {
#ifdef CUDNN
          cudaMemcpy(tnsr.data, input, sizeof(float)*len, cudaMemcpyHostToDevice);
#else
          printf("not define the macro CUDNN, but you are using the library, please define it in the Makefile\n");
          exit(1);
#endif
        }
        else if(tnsr.ctx.device_type==kOpenCL)
        {
#ifdef PERFDNN_OCL
          float *p=(float*)(tnsr.data);
          memcpy(p, input, sizeof(float)*len);

          perf_dnn_mat_t mat_cpu;
          if(tnsr.ndim==4)
          {
            mat_cpu = perf_dnn_init_mat_with_data((void*)input, tnsr.shape[2], tnsr.shape[3], tnsr.shape[3], tnsr.shape[1], tnsr.shape[0], PERFDNN_32F);
          }
          else if(tnsr.ndim==1)
          {
            mat_cpu = perf_dnn_init_mat_with_data((void*)input, 1, tnsr.shape[0], tnsr.shape[0], 1, 1, PERFDNN_32F);
          }
          else
          {
            printf("can not support the dimensions < 4\n");
            exit(1);
          }
          
          perf_dnn_upload_data_to_device(hd->perf_ocl_context, mat_cpu, tnsr.mat_ocl, 1, 0);
#else
          printf("not define the macro PERFDNN_OCL, but you are using the library, please define it in the Makefile\n");
          exit(1);
#endif
        }
        else
        {
          printf("can not support the device type %d in meta_float_to_tensor\n", tnsr.ctx.device_type);
        }
      }
      else
      {
        printf("unsupported metaformat, lanes:%d\n", tnsr.dtype.lanes);
      }
      break;
    }
    default: printf("unsupported metaformat, code:%d\n", tnsr.dtype.code);
  }

  return;
}
