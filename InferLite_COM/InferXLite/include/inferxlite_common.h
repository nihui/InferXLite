#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<stdint.h>
#include <stdbool.h>

#ifdef WIN64
//#include <winsock.h>
#include <winsock2.h>
#include <time.h>
#endif // WIN64


#ifdef WIN32
//#include <winsock.h>
#include <winsock2.h>
#include <time.h>

#endif
#ifdef LINUX
#include <sys/time.h>
#endif

#include "metafmt.h"



#ifdef CUDNN
#include "cuda.h"
#include "cudnn.h"
#endif

#ifdef PERFDNN_OCL
#include "core/core.h"
#include "ocl/ocl.h"
#endif

#ifndef INCLUDE_COMMON_H_
#define INCLUDE_COMMON_H_

#ifdef __cplusplus
extern "C"
{
#endif //cpp


/*
 * The device type in DLContext.
 */
typedef enum {
  kCPU = 1,
  kGPU = 2,
  // kCPUPinned = kCPU | kGPU
  kCPUPinned = 3,
  kOpenCL = 4,
  kMetal = 8,
  kVPI = 9,
  kROCM = 10,
} DLDeviceType;

/*
 * A Device context for Tensor and operator.
 */
typedef struct {
  /* The device type used in the device. */
  DLDeviceType device_type;
  /* The device index */
  int device_id;
} DLContext;

/*
 * The type code options DLDataType.
 */
typedef enum {
  kInt = 0U,
  kUInt = 1U,
  kFloat = 2U,
} DLDataTypeCode;

/*
 * The data type the tensor can hold.
 *
 *  Examples
 *   - float: type_code = 2, bits = 32, lanes=1
 *   - float4(vectorized 4 float): type_code = 2, bits = 32, lanes=4
 *   - int8: type_code = 0, bits = 8, lanes=1
 */
typedef struct {
  /*
   * Type code of base types.
   * We keep it uint8_t instead of DLDataTypeCode for minimal memory
   * footprint, but the value should be one of DLDataTypeCode enum values.
   * */
  uint8_t code;
  /*
   * Number of bits, common choices are 8, 16, 32.
   */
  uint8_t bits;
  /* Number of lanes in the type, used for vector types. */
  uint16_t lanes;
} DLDataType;




/*
 * Plain C Tensor object, does not manage memory.
 */
typedef struct {
  /*
   * The opaque data pointer points to the allocated data.
   *  This will be CUDA device pointer or cl_mem handle in OpenCL.
   *  This pointer is always aligns to 256 bytes as in CUDA.
   */
  void* data;

#ifdef PERFDNN_OCL
  perf_dnn_ocl_mat_t mat_ocl;
  perf_dnn_ocl_mat_t tmp_ocl;
#endif
  
  /* The device context of the tensor */
  DLContext ctx;
  /* Number of dimensions */
  int ndim;
  /* The data type of the pointer*/
  DLDataType dtype;
  /* The shape of the tensor */
  int* shape;
  /*
   * strides of the tensor,
   *  can be NULL, indicating tensor is compact.
   */
  int* strides;
  /* The offset in bytes to the beginning pointer to data */
  int byte_offset;

  void *desc;
} DLTensor;





enum CONV_WAY{conv_nature=0, conv_gemm=1, conv_perfdnn=2};

enum LRN_WAY{across_channels=0, within_channels=1};

enum OPERATION{SUM=0, PROD=1, MAX=2, AVE=3};

enum OPERATE{add=0, sub=1, mul=2, division=3};

enum UNIT_OP{convolution=0, linear=1, batchnorm=2, scale=3};


struct context_model
{
#ifdef CUDNN
  cudnnFilterDescriptor_t pFilterDesc;
  cudnnConvolutionDescriptor_t pConvDesc;
  int sizeInBytes;
  void* workSpace;
#endif
};

struct context_data
{
#ifdef CUDNN
  cudnnTensorDescriptor_t pDataDesc;
#endif
};



struct model_pipeline
{
  DLTensor *weight;
  int nw;
};

struct data_pipeline
{
  DLTensor datas;
};

struct data_arg
{
  enum UNIT_OP uo;
  int str_h;
  int str_w;
  int pad_h;
  int pad_w;
};


enum TIME_STAT_WAY{time_forward=0, time_layer=1, time_no=2};


struct TIMEBYLAYER
{
  double time;
  char *tname;
  struct TIMEBYLAYER *tp;
};

struct handler;
typedef void (*model_func_pointer)(char *path, char *model, char *data, void *pdata, void **pout, struct handler *hd);

struct func_pointer_map
{
  char *path;
  unsigned int len;
  char **name;
  model_func_pointer *func;
};



struct CONV_ARG
{
  int group;
  int input_n;
  int input_c;
  int input_h;
  int input_w;
  int kernel_h;
  int kernel_w;
  int output_c;
  int output_h;
  int output_w;
  int pad_h;
  int pad_w;
  int str_h;
  int str_w;
  int dila_h;
  int dila_w;
};





struct handler
{
  unsigned int max_num_pipeline;
  struct model_pipeline *modelflow;
  struct data_pipeline *dataflow;
  bool weight_has_load;
  bool data_has_init;
  bool is_update_input;
  enum TIME_STAT_WAY tsw;
  struct TIMEBYLAYER *time_head;
  struct TIMEBYLAYER *time_tail;
  int time_layer_cnt;
  struct timeval start_forward, start_layer;
  uint64_t *elem;
  enum CONV_WAY cw;
  DLDeviceType dvct;
  struct func_pointer_map fpm;
  unsigned int max_num_func;
  char **is_has_init;
  unsigned int len_init;
  unsigned int len_elem;
#ifdef CUDNN
  cudnnHandle_t hCudNN;
#endif
#ifdef PERFDNN_OCL
  perf_dnn_ocl_context_t perf_ocl_context;
#endif
};






typedef struct
{
  model_func_pointer func;
  char *model;
  char *data;
  int tag;
  struct handler hd;
}inferx_context;

#ifdef __cplusplus
}
#endif //cpp

#endif //INCLUDE_COMMON_H_
