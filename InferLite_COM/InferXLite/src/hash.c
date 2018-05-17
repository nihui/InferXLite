#include "hash.h"

struct data_pipeline* data_map(const char *p, struct handler *hd)
{
  uint64_t value=hash_str(p, strlen(p));
  int index=hash_long_get(value, hd);

  if(index==-1)
  {
    printf("the data %s should exist, but no\n", p);
    exit(1);
  }

  return hd->dataflow+index;
}


struct data_pipeline* data_map_init(const char *p, int *nchw, struct handler *hd)
{
  uint64_t value=hash_str(p, strlen(p));
  int index=hash_long_insert(value, hd);

  dltensor_type_init(&(hd->dataflow[index].datas), 2, 32, 1);

  hd->dataflow[index].datas.ndim = 4;
  hd->dataflow[index].datas.shape=(int*)malloc(sizeof(int)*hd->dataflow[index].datas.ndim);

  hd->dataflow[index].datas.ctx.device_type = hd->dvct;

  int j=0;
  for(j=0; j<4; j++)
  {
    hd->dataflow[index].datas.shape[j]=nchw[j];
  }

  long len=1;
  for(j=0; j<hd->dataflow[index].datas.ndim; j++)
    len*=hd->dataflow[index].datas.shape[j];
  len*=hd->dataflow[index].datas.dtype.bits/8*hd->dataflow[index].datas.dtype.lanes;

  if(hd->dataflow[index].datas.ctx.device_type==kCPU)
  {
    hd->dataflow[index].datas.data=(void*)malloc(len);
    memset(hd->dataflow[index].datas.data, 0, len);
  }
  else if(hd->dataflow[index].datas.ctx.device_type==kGPU)
  {
#ifdef CUDNN
    cudaError_t err;
    float *p;
    err = cudaMalloc((void**)&p, len);
    if (err != cudaSuccess)
      printf("cudnn error %d\n", err);
    hd->dataflow[index].datas.data = (void*)p;

    struct context_data *desc = (struct context_data*)(hd->dataflow[index].datas.desc);

    cudnnStatus_t status;
    status = cudnnCreateTensorDescriptor(&(desc->pDataDesc));
    if (status != CUDNN_STATUS_SUCCESS)
      printf("cudnn error %d\n", err);

    cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
    status = cudnnSetTensor4dDescriptor(desc->pDataDesc, CUDNN_TENSOR_NCHW, dataType, nchw[0], nchw[1], nchw[2], nchw[3]);
    if(status != CUDNN_STATUS_SUCCESS)
      printf("cudnn error %d\n", err);
#else
    printf("not define the macro CUDNN, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else if(hd->dataflow[index].datas.ctx.device_type==kOpenCL)
  {
#ifdef PERFDNN_OCL
    hd->dataflow[index].datas.data=(void*)malloc(len);
    memset(hd->dataflow[index].datas.data, 0, len);
    hd->dataflow[index].datas.mat_ocl = perf_dnn_init_ocl_mat(hd->perf_ocl_context, nchw[2], nchw[3], nchw[3], nchw[1], nchw[0], PERFDNN_32F,0,NULL);
#else
    printf("not define the macro PERFDNN_OCL, but you are using the library, please define it in the Makefile\n");
    exit(1);
#endif
  }
  else
  {
    printf("can not support the device type %d in data_map_init\n", hd->dataflow[index].datas.ctx.device_type);
  }

  return hd->dataflow+index;
}


struct model_pipeline* weight_bias_map(const char *p, struct handler *hd)
{
  uint64_t value=hash_str(p, strlen(p));
  int index=hash_long_get(value, hd);
  if(index==-1)
  {
    printf("the model %s read from other software is different from our definition\n", p);
    exit(1);
  }

  return hd->modelflow+index;
}


struct model_pipeline* weight_bias_map_init(const char *p, int nw, int *nv, int *vw, struct data_arg dg, struct handler *hd)
{
  uint64_t value=hash_str(p, strlen(p));
  int index=hash_long_insert(value, hd);

  hd->modelflow[index].nw=nw;
  hd->modelflow[index].weight=(DLTensor *)malloc(sizeof(DLTensor)*nw);

  int i, j;
  int vw_i=0;
  for(i=0; i<nw; i++)
  {
    dltensor_type_init(&(hd->modelflow[index].weight[i]), 2, 32, 1);

    hd->modelflow[index].weight[i].ndim=nv[i];
    hd->modelflow[index].weight[i].shape=(int*)malloc(sizeof(int)*hd->modelflow[index].weight[i].ndim);

    hd->modelflow[index].weight[i].ctx.device_type = hd->dvct;

    for(j=0; j<nv[i]; j++)
    {
      hd->modelflow[index].weight[i].shape[j]=vw[vw_i++];
    }
    long len=1;
    for(j=0; j<hd->modelflow[index].weight[i].ndim; j++)
      len*=hd->modelflow[index].weight[i].shape[j];
    len*=hd->modelflow[index].weight[i].dtype.bits/8*hd->modelflow[index].weight[i].dtype.lanes;


    if(hd->modelflow[index].weight[i].ctx.device_type==kCPU)
    {
      hd->modelflow[index].weight[i].data=(void*)malloc(len);
      memset(hd->modelflow[index].weight[i].data, 0, len);
    }
    else if(hd->modelflow[index].weight[i].ctx.device_type==kGPU)
    {
#ifdef CUDNN
      cudaError_t err;
      float *p;
      err = cudaMalloc((void**)&p, len);
      if (err != cudaSuccess)
        printf("cudnn error %d\n", err);
      hd->modelflow[index].weight[i].data = (void*)p;
#else
      printf("not define the macro CUDNN, but you are using the library, please define it in the Makefile\n");
      exit(1);
#endif
    }
    else if(hd->modelflow[index].weight[i].ctx.device_type==kOpenCL)
    {
#ifdef PERFDNN_OCL
      hd->modelflow[index].weight[i].data=(void*)malloc(len);
      memset(hd->modelflow[index].weight[i].data, 0, len);
      long nchw_new[4];
      int nn; 
      for(nn=0; nn<4; nn++)
      {
        nchw_new[nn]=1;
      }
      if(hd->modelflow[index].weight[i].ndim<=4)
      {
        for(nn=0; nn<hd->modelflow[index].weight[i].ndim; nn++)
          nchw_new[nn]=hd->modelflow[index].weight[i].shape[nn];
      }
      else
      {
        printf("the perf dnn ocl can not support the dimensions > 5\n");
        exit(1);
      }
      hd->modelflow[index].weight[i].mat_ocl = perf_dnn_init_ocl_mat(hd->perf_ocl_context, nchw_new[0], nchw_new[2]*nchw_new[3]*nchw_new[1], nchw_new[2]*nchw_new[3]*nchw_new[1], 1, 1, PERFDNN_32F,0,NULL);
      hd->modelflow[index].weight[i].tmp_ocl = NULL;
#else
      printf("not define the macro PERFDNN_OCL, but you are using the library, please define it in the Makefile\n");
      exit(1);
#endif
    }
    else
    {
      printf("can not support the device type %d in weight_bias_map_init\n", hd->modelflow[index].weight[i].ctx.device_type);
    }
  }

#ifdef CUDNN
  //convolution
  if(dg.uo==convolution)
  {
    if(hd->modelflow[index].weight[0].ctx.device_type==kGPU)
    {
      struct context_model *desc = (struct context_model*)(hd->modelflow[index].weight[0].desc);

      cudnnStatus_t status;

      status = cudnnCreateFilterDescriptor(&(desc->pFilterDesc));
      if (status != CUDNN_STATUS_SUCCESS)
        printf("cudnn error %d\n", status);
      
      status = cudnnCreateConvolutionDescriptor(&(desc->pConvDesc));
      if (status != CUDNN_STATUS_SUCCESS)
        printf("cudnn error %d\n", status);

	  #define tensorDims 4
      int filter_dimA[tensorDims];
      filter_dimA[0] = hd->modelflow[index].weight[0].shape[0];
      filter_dimA[1] = hd->modelflow[index].weight[0].shape[1];
      filter_dimA[2] = hd->modelflow[index].weight[0].shape[2];
      filter_dimA[3] = hd->modelflow[index].weight[0].shape[3];

      status = cudnnSetFilterNdDescriptor(desc->pFilterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, tensorDims, filter_dimA);
      if (status != CUDNN_STATUS_SUCCESS)
        printf("cudnn error %d\n", status);

#define convDims 2
      int padA[convDims];
      padA[0] = dg.pad_h;
      padA[1] = dg.pad_w;
      int filterStrideA[convDims];
      filterStrideA[0] = dg.str_h;
      filterStrideA[1] = dg.str_w;
      int upscaleA[convDims];
      upscaleA[0] = 1;
      upscaleA[1] = 1;
      status = cudnnSetConvolutionNdDescriptor(desc->pConvDesc, convDims, padA, filterStrideA, upscaleA, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);
      if (status != CUDNN_STATUS_SUCCESS)
        printf("cudnn error %d\n", status);
    }
  }
#endif

  return hd->modelflow+index;
}



uint64_t hash_str(const char* p, int n) {
    if (n == 0 || NULL == p) {
        return 0;
    }

    uint64_t h = 2654435761u * (*p);
    //printf("%c\n", *p);

    while (--n) {
        h = (h * 97) + *++p;
    }

    if(h==0)
    {
      printf("can not support the 0-hash-value mode\n");
      exit(1);
    }

    return h;
}




int hash_long_insert(const uint64_t n_hash, struct handler *hd)
{
    if(hd->elem==NULL)
    {
      hd->len_elem=hd->max_num_pipeline;
      hd->elem=(uint64_t*)malloc(sizeof(uint64_t)*hd->len_elem);
      memset(hd->elem, 0, sizeof(uint64_t)*hd->len_elem);
    }
    int n_hash_start = n_hash % hd->len_elem;
    int n_hash_pos = n_hash_start;

    while(hd->elem[n_hash_pos])
    {
       if (hd->elem[n_hash_pos] == n_hash)
       {
         break;
       }
       else
       {
         n_hash_pos = (n_hash_pos + 1) % hd->len_elem;
         if(n_hash_pos == n_hash_start)
         {
           printf("no space in the precious hash table len(%d) pos->%d to insert the string: %ld !!!\n", hd->len_elem, n_hash_pos, n_hash);  
           exit(1);
         }
       }
    }

    if(!hd->elem[n_hash_pos])
    {
      hd->elem[n_hash_pos]=n_hash;
    }

    return n_hash_pos;
}


int hash_long_get(const uint64_t n_hash, struct handler *hd)
{
    int n_hash_start = n_hash % hd->len_elem;
    int n_hash_pos = n_hash_start;

    while(hd->elem[n_hash_pos])
    {
       if (hd->elem[n_hash_pos] == n_hash)
       {
         return n_hash_pos;
       }
       else
       {
         n_hash_pos = (n_hash_pos + 1) % hd->len_elem;
       }
       if(n_hash_pos == n_hash_start)
       {
         break;
       }
    }

    return -1;
}

//this is for debug
void print_elem_value(int index, struct handler *hd)
{
  printf("elem[%d]=%ld\n", index, hd->elem[index]);

  return;
}


void dltensor_type_init(DLTensor *tsr, int code, int bits, int lanes)
{
  tsr->dtype.code = code;
  tsr->dtype.bits = bits;
  tsr->dtype.lanes = lanes;
  
  return;
}
