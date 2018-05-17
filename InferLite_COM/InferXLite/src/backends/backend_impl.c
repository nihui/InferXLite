#include "backend_impl.h"
#include "math.h"

#ifdef BLAS
#include "cblas.h"
#endif

#ifdef CUDNN
#include "cuda.h"
#include "cudnn.h"
#endif


#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON



#define ACCESS_INPUT_NCHW(i, j, k, v) input[((i * conv_arg_pt->input_c + j) * conv_arg_pt->input_h + k) * conv_arg_pt->input_w + v]
#define ACCESS_OUTPUT_NKPQ(i, j, k, v) output[((i * conv_arg_pt->output_c + j) * conv_arg_pt->output_h + k) * conv_arg_pt->output_w + v]
#define ACCESS_FILTER_KCRS(i, j, k, v) filter[((i * conv_arg_pt->input_c + j) * conv_arg_pt->kernel_h + k) * conv_arg_pt->kernel_w + v]

#define ACCESS_INPUT_NHWC(i, j, k, v) input[((i * conv_arg_pt->input_h + j) * conv_arg_pt->input_w + k) * conv_arg_pt->input_c + v]
#define ACCESS_OUTPUT_NPQK(i, j, k, v) output[((i * conv_arg_pt->output_h + j) * conv_arg_pt->output_w + k) * conv_arg_pt->output_c + v]
#define ACCESS_FILTER_RSCK(i, j, k, v) filter[((i * conv_arg_pt->kernel_w + j) * conv_arg_pt->input_c + k) * conv_arg_pt->output_c + v]

#define ADDRESS_OUTPUT_NPQK(i, j, k, v) (output + ((i * conv_arg_pt->output_h + j) * conv_arg_pt->output_w + k) * conv_arg_pt->output_c + v)
#define ADDRESS_FILTER_RSCK(i, j, k, v) (filter + ((i * conv_arg_pt->kernel_w + j) * conv_arg_pt->input_c + k) * conv_arg_pt->output_c + v)


#define MAX(a,b) (a>b?a:b)
#define MIN(a,b) (a<b?a:b)



#include <time.h>
#ifdef WIN32
#   include <windows.h>
#else
#   include <sys/time.h>
#endif

#ifdef WIN32
int
gettimeofday(struct timeval *tp, void *tzp)
{
	time_t clock;
	struct tm tm;
	SYSTEMTIME wtm;

	GetLocalTime(&wtm);
	tm.tm_year = wtm.wYear - 1900;
	tm.tm_mon = wtm.wMonth - 1;
	tm.tm_mday = wtm.wDay;
	tm.tm_hour = wtm.wHour;
	tm.tm_min = wtm.wMinute;
	tm.tm_sec = wtm.wSecond;
	tm.tm_isdst = -1;
	clock = mktime(&tm);
	tp->tv_sec = clock;
	tp->tv_usec = wtm.wMilliseconds * 1000;

	return (0);
}
#endif

static inline float safe_log(float input)
{
	return log(input > exp(-50.0) ? input : exp(-50.0));
}

void convolution_forward_naive_no_pad_nchw_impl(struct CONV_ARG *conv_arg_pt, float *input, float *filter, float *output)
{
  int n, k, c, p, q, r, s;

  for(n = 0; n < conv_arg_pt->input_n; ++n)
  {
    for(k = 0; k < conv_arg_pt->output_c/conv_arg_pt->group; ++k)
    {
      for(c = 0; c < conv_arg_pt->input_c/conv_arg_pt->group; ++c)
      {
        for(p = 0; p < conv_arg_pt->output_h; ++p)
        {
          int ih = p * conv_arg_pt->str_h;
          for(q = 0; q < conv_arg_pt->output_w; ++q)
          {
            int iw = q * conv_arg_pt->str_w;
            for(r = 0; r < conv_arg_pt->kernel_h; ++r)
            {
              for(s = 0; s < conv_arg_pt->kernel_w; ++s)
              {
                ACCESS_OUTPUT_NKPQ(n, k, p, q) += ACCESS_INPUT_NCHW(n, c, (ih + r), (iw + s)) * ACCESS_FILTER_KCRS(k, c, r, s); 
              }
            }
          }
        }
      }
    }
  }

  return;
}


void convolution_forward_naive_pad_nchw_impl(struct CONV_ARG *conv_arg_pt, float *input, float *filter, float *output)
{
  int n, k, c, p, q;

  for(n = 0; n < conv_arg_pt->input_n; ++n)
  {
    for(k = 0; k < conv_arg_pt->output_c/conv_arg_pt->group; ++k)
    {
      for(c = 0; c < conv_arg_pt->input_c/conv_arg_pt->group; ++c)
      {
        for(p = 0; p < conv_arg_pt->output_h; ++p)
        {
          int ih = p * conv_arg_pt->str_h - conv_arg_pt->pad_h;
          for(q = 0; q < conv_arg_pt->output_w; ++q)
          {
            int iw = q * conv_arg_pt->str_w - conv_arg_pt->pad_w;
            int r_end = ih + conv_arg_pt->kernel_h < conv_arg_pt->input_h ?  conv_arg_pt->kernel_h : conv_arg_pt->input_h - ih;
            int s_end = iw + conv_arg_pt->kernel_w < conv_arg_pt->input_w ?  conv_arg_pt->kernel_w : conv_arg_pt->input_w - iw;
            int r = ih < 0 ? -ih : 0;
            for(; r < r_end; ++r)
            {
              int s = iw < 0 ? -iw : 0;
              for(; s < s_end; ++s)
              {
                ACCESS_OUTPUT_NKPQ(n, k, p, q) += ACCESS_INPUT_NCHW(n, c, (ih + r), (iw + s)) *
                  ACCESS_FILTER_KCRS(k, c, r, s); 
              }
            }
          }
        }
      }
    }
  }

  return;
}




void convolution_forward_naive_pad_nhwc_impl(struct CONV_ARG *conv_arg_pt, float *input, float *filter, float *output)
{
  int n, p, q, k, c;

  for (n = 0; n < conv_arg_pt->input_n; ++n)
  {
    for (p = 0; p < conv_arg_pt->output_h; ++p)
    {
      int ih = p * conv_arg_pt->str_h - conv_arg_pt->pad_h;
      for (q = 0; q < conv_arg_pt->output_w; ++q)
      {
        int iw = q * conv_arg_pt->str_w - conv_arg_pt->pad_w;
        int r_end = ih + conv_arg_pt->kernel_h < conv_arg_pt->input_h ? conv_arg_pt->kernel_h : conv_arg_pt->input_h - ih;
        int s_end = iw + conv_arg_pt->kernel_w < conv_arg_pt->input_w ? conv_arg_pt->kernel_w : conv_arg_pt->input_w - iw;
        int r = ih < 0 ? -ih : 0;
        for (; r < r_end; ++r)
        {
          int s = iw < 0 ? -iw : 0;
          for (; s < s_end; ++s)
          {
            for (k = 0; k < conv_arg_pt->output_c; ++k)
            {
              for (c = 0; c < conv_arg_pt->input_c; ++c)
              {
                ACCESS_OUTPUT_NPQK(n, p, q, k) += ACCESS_INPUT_NHWC(n, (ih + r), (iw + s), c) *
                  ACCESS_FILTER_RSCK(r, s, c, k); 
              }
            }
          }
        }
      }
    }
  }

  return;
}




void convolution_forward_perfdnn_nchw_impl(struct CONV_ARG *conv_arg_pt, float *input, float *filter, float *bias, float *output, bool *is_bias,int activation_type)
{
#ifdef PERFDNN
  enum perfdnn_status init_status = perfdnn_initialize();

  enum perfdnn_convolution_algorithm algorithm = perfdnn_convolution_algorithm_auto;
  //enum perfdnn_convolution_algorithm algorithm = perfdnn_convolution_algorithm_wt8x8;
  //enum perfdnn_convolution_algorithm algorithm = perfdnn_convolution_algorithm_im2col_gemm;
  //enum perfdnn_convolution_algorithm algorithm = perfdnn_convolution_algorithm_ft8x8;
  //enum perfdnn_convolution_algorithm algorithm = perfdnn_convolution_algorithm_ft16x16;
  const struct perfdnn_size output_subsampling = {.width=conv_arg_pt->str_w, .height=conv_arg_pt->str_h};

  struct perfdnn_size input_size;
  input_size.width = conv_arg_pt->input_w;
  input_size.height = conv_arg_pt->input_h;

  struct perfdnn_padding input_padding;
  input_padding.top = input_padding.bottom = conv_arg_pt->pad_h;
  input_padding.left = input_padding.right = conv_arg_pt->pad_w;

  struct perfdnn_size kernel_size;
  kernel_size.width = conv_arg_pt->kernel_w;
  kernel_size.height = conv_arg_pt->kernel_h;

  struct perfdnn_size dilation_size;
  dilation_size.width = conv_arg_pt->dila_w;
  dilation_size.height = conv_arg_pt->dila_h;

  if(activation_type == 0)
  {
    perfdnn_convolution_inference(algorithm, conv_arg_pt->input_c, conv_arg_pt->output_c, input_size, input_padding, kernel_size, output_subsampling, dilation_size, input, filter, bias, output, conv_arg_pt->group, *is_bias, perfdnn_activation_identity);
  }
  else if(activation_type == 1)
  {
    perfdnn_convolution_inference(algorithm, conv_arg_pt->input_c, conv_arg_pt->output_c, input_size, input_padding, kernel_size, output_subsampling, dilation_size, input, filter, bias, output, conv_arg_pt->group, *is_bias, perfdnn_activation_relu);
  }

  *is_bias=false;

#else
  printf("not define the macro PERFDNN, but you are using the library, please define it in the Makefile\n");
  exit(1);
#endif

  return;
}






void convolution_forward_gemm_nchw_impl(struct CONV_ARG *conv_arg_pt, float *input, float *filter, float *output)
{
  int unpack_size=conv_arg_pt->input_c*conv_arg_pt->kernel_h*conv_arg_pt->kernel_w*conv_arg_pt->output_h*conv_arg_pt->output_w;
  float *unpack=(float*)malloc(unpack_size*sizeof(float));
  memset(unpack, 0, unpack_size*sizeof(float));
  unpack_dilation_nchw_impl(conv_arg_pt, input, unpack);

  int ilen=0, olen=0, wlen=0;

  if(conv_arg_pt->group>1)
  {
    ilen=conv_arg_pt->input_c*conv_arg_pt->kernel_h*conv_arg_pt->kernel_w*conv_arg_pt->output_h*conv_arg_pt->output_w/conv_arg_pt->group;
    olen=conv_arg_pt->output_c*conv_arg_pt->output_h*conv_arg_pt->output_w/conv_arg_pt->group;
    wlen=conv_arg_pt->input_c*conv_arg_pt->output_c*conv_arg_pt->kernel_h*conv_arg_pt->kernel_w/conv_arg_pt->group/conv_arg_pt->group;
  }

  for(int g=0; g<conv_arg_pt->group; g++)
  {
    gemm(filter+g*wlen, unpack+g*ilen, output+g*olen, false, false, 1.0, 0.0, conv_arg_pt->output_c/conv_arg_pt->group, conv_arg_pt->output_h*conv_arg_pt->output_w, conv_arg_pt->input_c*conv_arg_pt->kernel_h*conv_arg_pt->kernel_w/conv_arg_pt->group);
  }
  free((void*)unpack);

  return;
}




void convolution_backward_gemm_nchw_impl(struct CONV_ARG *conv_arg_pt, float *input, float *filter, float *output)
{
  int pack_size=conv_arg_pt->input_c*conv_arg_pt->kernel_h*conv_arg_pt->kernel_w*conv_arg_pt->output_h*conv_arg_pt->output_w;
  float *pack=(float*)malloc(pack_size*sizeof(float));
  memset(pack, 0, pack_size*sizeof(float));
  gemm(filter, input, pack, true, false, 1.0, 0.0, conv_arg_pt->input_c*conv_arg_pt->kernel_h*conv_arg_pt->kernel_w, conv_arg_pt->output_h*conv_arg_pt->output_w, conv_arg_pt->output_c);
  pack_nchw_impl(conv_arg_pt, pack, output);
  free((void*)pack);

  return;
}







void convolution_backward_naive_no_pad_nchw_impl(struct CONV_ARG *conv_arg_pt, float *input, float *filter, float *output)
{
  int n, c, k, p, q, r, s;

  if (conv_arg_pt->pad_h == 0 && conv_arg_pt->pad_w == 0)
  {
    for (n = 0; n < conv_arg_pt->input_n; ++n)
    {
      for (c = 0; c < conv_arg_pt->input_c; ++c)
      {
        for (k = 0; k < conv_arg_pt->output_c; ++k)
        {
          for (p = 0; p < conv_arg_pt->output_h; ++p)
          {
            int ih = p * conv_arg_pt->str_h;
            for (q = 0; q < conv_arg_pt->output_w; ++q)
            {
              int iw = q * conv_arg_pt->str_w;
              for (r = 0; r < conv_arg_pt->kernel_h; ++r)
              {
                for (s = 0; s < conv_arg_pt->kernel_w; ++s)
                {
                  ACCESS_INPUT_NCHW(n, c, (ih + r), (iw + s)) += ACCESS_OUTPUT_NKPQ(n, k, p, q) *
                    ACCESS_FILTER_KCRS(k, c, r, s); 
                }
              }
            }
          }
        }
      }
    }
  }

  return;
}




void convolution_backward_naive_pad_nchw_impl(struct CONV_ARG *conv_arg_pt, float *input, float *filter, float *output)
{
  int n, c, k, p, q;

  for (n = 0; n < conv_arg_pt->input_n; ++n)
  {
    for (c = 0; c < conv_arg_pt->input_c; ++c)
    {
      for (k = 0; k < conv_arg_pt->output_c; ++k)
      {
        for (p = 0; p < conv_arg_pt->output_h; ++p)
        {
          int ih = p * conv_arg_pt->str_h - conv_arg_pt->pad_h;
          for (q = 0; q < conv_arg_pt->output_w; ++q)
          {
            int iw = q * conv_arg_pt->str_w - conv_arg_pt->pad_w;
            int r_end = ih + conv_arg_pt->kernel_h < conv_arg_pt->input_h ? conv_arg_pt->kernel_h : conv_arg_pt->input_h - ih;
            int s_end = iw + conv_arg_pt->kernel_w < conv_arg_pt->input_w ? conv_arg_pt->kernel_w : conv_arg_pt->input_w - iw;
            int r = ih < 0 ? -ih : 0;
            for (; r < r_end; ++r)
            {
              int s = iw < 0 ? -iw : 0;
              for (; s < s_end; ++s)
              {
                ACCESS_INPUT_NCHW(n, c, (ih + r), (iw + s)) += ACCESS_OUTPUT_NKPQ(n, k, p, q) *
                  ACCESS_FILTER_KCRS(k, c, r, s); 
              }
            }
          }
        }
      }
    }
  }

  return;
}


void convolution_backward_naive_pad_nhwc_impl(struct CONV_ARG *conv_arg_pt, float *input, float *filter, float *output)
{
  int n, p, q, k, c;

  for (n = 0; n < conv_arg_pt->input_n; ++n)
  {
    for (p = 0; p < conv_arg_pt->output_h; ++p)
    {
      int ih = p * conv_arg_pt->str_h - conv_arg_pt->pad_h;
      for (q = 0; q < conv_arg_pt->output_w; ++q)
      {
        int iw = q * conv_arg_pt->str_w - conv_arg_pt->pad_w;
        int r_end = ih + conv_arg_pt->kernel_h < conv_arg_pt->input_h ? conv_arg_pt->kernel_h : conv_arg_pt->input_h - ih;
        int s_end = iw + conv_arg_pt->kernel_w < conv_arg_pt->input_w ? conv_arg_pt->kernel_w : conv_arg_pt->input_w - iw;
        int r = ih < 0 ? -ih : 0;
        for (; r < r_end; ++r)
        {
          int s = iw < 0 ? -iw : 0;
          for (; s < s_end; ++s)
          {
            for (k = 0; k < conv_arg_pt->output_c; ++k)
            {
              for (c = 0; c < conv_arg_pt->input_c; ++c)
              {
                ACCESS_INPUT_NHWC(n, (ih + r), (iw + s), c) += ACCESS_OUTPUT_NPQK(n, p, q, k) *
                  ACCESS_FILTER_RSCK(r, s, c, k);
              }
            }
          }
        }
      }
    }
  }

  return;
}


void max_pooling_forward_nchw_impl(struct CONV_ARG *conv_arg_pt, float *input, float *output)
{
  // offset
  const int input_hw = conv_arg_pt->input_h * conv_arg_pt->input_w;
  const int input_chw = conv_arg_pt->input_c * input_hw;
  const int output_hw = conv_arg_pt->output_h * conv_arg_pt->output_w;
  const int output_chw = conv_arg_pt->output_c * output_hw;

  int n, c, oh, ow, h, w;
  for (n = 0; n < conv_arg_pt->input_n; ++n)
  {
    for (c = 0; c < conv_arg_pt->input_c; ++c)
    {
      const float* input_slice = input + n * input_chw + c * input_hw;
      float* output_slice = output + n * output_chw + c * output_hw;
      for (oh = 0; oh < conv_arg_pt->output_h; ++oh)
      {
        for (ow = 0; ow < conv_arg_pt->output_w; ++ow)
        {
          int hs = oh * conv_arg_pt->str_h-conv_arg_pt->pad_h;
          int ws = ow * conv_arg_pt->str_w-conv_arg_pt->pad_w;
          int he = MIN(hs + conv_arg_pt->kernel_h, conv_arg_pt->input_h);
          int we = MIN(ws + conv_arg_pt->kernel_w, conv_arg_pt->input_w);
          hs = MAX(hs, 0);
          ws = MAX(ws, 0);
          int pool_index = oh * conv_arg_pt->output_w + ow;
          int max_index_slice = hs * conv_arg_pt->input_w + ws;
          for (h = hs; h < he; ++h)
          {
            for (w = ws; w < we; ++w)
            {
              int index = h * conv_arg_pt->input_w + w;
              if (input_slice[index] > input_slice[max_index_slice])
              {
                max_index_slice = index;
              }
            }
          }
          output_slice[pool_index] = input_slice[max_index_slice];
        }
      }
    }
  }

  return;
}



void max_pooling_forward_nc3x3s2_impl(struct CONV_ARG *conv_arg_pt, float *input, float *output)
{
  //count time 
  struct timeval start;
  // offset
  const int input_hw = conv_arg_pt->input_h * conv_arg_pt->input_w;
  const int input_chw = conv_arg_pt->input_c * input_hw;
  const int output_hw = conv_arg_pt->output_h * conv_arg_pt->output_w;
  const int output_chw = conv_arg_pt->output_c * output_hw;

  //printf("input height %d, width %d\n",conv_arg_pt->input_h,conv_arg_pt->input_w);
  //printf("output height %d, width %d\n",conv_arg_pt->output_h,conv_arg_pt->output_w);
  const int tailstep = conv_arg_pt->input_w - 2*conv_arg_pt->output_w + conv_arg_pt->input_w;
//  const int tailstep =  conv_arg_pt->input_w;
  int n, c, oh;
  for (n = 0; n < conv_arg_pt->input_n; ++n)
  {
    gettimeofday(&start, NULL);
    for (c = 0; c < conv_arg_pt->input_c; ++c)
    {
      const float* input_slice = input + n * input_chw + c * input_hw;
      float* output_slice = output + n * output_chw + c * output_hw;
      const float * r0 = input_slice;
      const float * r1 = input_slice + conv_arg_pt->input_w;
      const float * r2 = input_slice + conv_arg_pt->input_w*2;
      for (oh = 0; oh < conv_arg_pt->output_h; ++oh){
#if __ARM_NEON
#if __aarch64__
        int num = conv_arg_pt->output_w >> 2 ;
        int left = conv_arg_pt->output_w - (num << 2);
        //printf("left %d\n",left);

#else
	int left =conv_arg_pt->output_w;
#endif
#else
        int left = conv_arg_pt->output_w;
#endif //__ARM_NEON

#if __ARM_NEON
#if __aarch64__
          float32x4x2_t _r0 = vld2q_f32(r0);
          float32x4x2_t _r1 = vld2q_f32(r1);
          float32x4x2_t _r2 = vld2q_f32(r2);
        for (; num>1; num--)
        {
                float32x4x2_t _r0n = vld2q_f32(r0+2);
                float32x4x2_t _r1n = vld2q_f32(r1+2);
                float32x4x2_t _r2n = vld2q_f32(r2+2);

                float32x4_t _max0 = vmaxq_f32(_r0.val[0], _r0.val[1]);
                float32x4_t _max1 = vmaxq_f32(_r1.val[0], _r1.val[1]);
                float32x4_t _max2 = vmaxq_f32(_r2.val[0], _r2.val[1]);

                float32x4_t _r02 = _r0n.val[0];
                float32x4_t _r12 = _r1n.val[0];
                float32x4_t _r22 = _r2n.val[0];

                _max0 = vmaxq_f32(_max0, _r02);
                _max1 = vmaxq_f32(_max1, _r12);
                _max2 = vmaxq_f32(_max2, _r22);
                
                float32x4_t _max;
                if(oh ==(conv_arg_pt->output_h-1) && (2*oh) == (conv_arg_pt->output_w-2))
                {
		  _max = vmaxq_f32(_max0, _max1);
                }
                else
                {  
                   _max = vmaxq_f32(vmaxq_f32(_max0, _max1), _max2);
                }
                vst1q_f32(output_slice, _max);

                _r0 = vld2q_f32(r0+8);
                _r1 = vld2q_f32(r1+8);
                _r2 = vld2q_f32(r2+8);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                /* 
                printf("output position: %d, %d, %d, pooling %lf\n",c,oh,(Q-nn*4)-left, *output_slice);
                printf("output position: %d, %d, %d, pooling %lf\n",c,oh,(Q-nn*4)+1-left, *(output_slice+1));
                printf("output position: %d, %d, %d, pooling %lf\n",c,oh,(Q-nn*4)+2-left, *(output_slice+2));
                printf("output position: %d, %d, %d, pooling %lf\n",c,oh,(Q-nn*4)+3-left, *(output_slice+3));
                */
                output_slice += 4;

        }
        for (; num>0; num--)
        {
                
                float32x4x2_t _r0n = vld2q_f32(r0+2);
                float32x4x2_t _r1n = vld2q_f32(r1+2);
                float32x4x2_t _r2n = vld2q_f32(r2+2);

                float32x4_t _max0 = vmaxq_f32(_r0.val[0], _r0.val[1]);
                float32x4_t _max1 = vmaxq_f32(_r1.val[0], _r1.val[1]);
                float32x4_t _max2 = vmaxq_f32(_r2.val[0], _r2.val[1]);

                float32x4_t _r02 = _r0n.val[0];
                float32x4_t _r12 = _r1n.val[0];
                float32x4_t _r22 = _r2n.val[0];
                if(conv_arg_pt->input_w==2*conv_arg_pt->output_w && left==0)
                {
                  float32x4_t zero=vdupq_n_f32(0.0);
                  
                  _r02 = vextq_f32(_r0.val[0], zero, 1);
                  _r12 = vextq_f32(_r1.val[0], zero, 1);
                  _r22 = vextq_f32(_r2.val[0], zero, 1);
                }  
                  _max0 = vmaxq_f32(_max0, _r02);
                  _max1 = vmaxq_f32(_max1, _r12);
                  _max2 = vmaxq_f32(_max2, _r22);

                float32x4_t _max;
                if(oh ==(conv_arg_pt->output_h-1) && (2*oh) == (conv_arg_pt->input_h-2))
                {
		  _max = vmaxq_f32(_max0, _max1);
                }
                else
                {  
                   _max = vmaxq_f32(vmaxq_f32(_max0, _max1), _max2);
                }
                vst1q_f32(output_slice, _max);

                _r0 = vld2q_f32(r0+8);
                _r1 = vld2q_f32(r1+8);
                _r2 = vld2q_f32(r2+8);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                /* 
                printf("output position: %d, %d, %d, pooling %lf\n",c,oh,(Q-nn*4)-left, *output_slice);
                printf("output position: %d, %d, %d, pooling %lf\n",c,oh,(Q-nn*4)+1-left, *(output_slice+1));
                printf("output position: %d, %d, %d, pooling %lf\n",c,oh,(Q-nn*4)+2-left, *(output_slice+2));
                printf("output position: %d, %d, %d, pooling %lf\n",c,oh,(Q-nn*4)+3-left, *(output_slice+3));
                */
                output_slice += 4;

        }
#endif
#endif // __ARM_NEON
       // int x,y,z;
       // x=1;
       // y=2;
       // z=3;
        for(;left>0; left--)
          {
            float max0,max1,max2;
            max0=0.0;
            max1=0.0;
            max2=0.0;
             
            if(left ==1 &&(2*(conv_arg_pt->output_w-1))==(conv_arg_pt->input_w-2))
            {
             //printf("x,y,z %d,%d,%d\n", x,y,z);
             //printf("the last pooling\n");
             max0 = MAX(r0[0], r0[1]);
             max1 = MAX(r1[0], r1[1]);
             max2 = MAX(r2[0], r2[1]);
		
	    }
            else
            {
             max0 = MAX(MAX(r0[0], r0[1]), r0[2]);
             max1 = MAX(MAX(r1[0], r1[1]), r1[2]);
             max2 = MAX(MAX(r2[0], r2[1]), r2[2]);
            }
            if(oh ==(conv_arg_pt->output_h-1) && (2*oh) == (conv_arg_pt->input_h-2))
            {
              *output_slice = MAX(max0, max1);
            }
            else
	    {
              *output_slice = MAX(MAX(max0, max1), max2);
            }
	    // printf("output position: %d, %d, %d, pooling %lf\n",c,oh,Q-left, *output_slice);
            r0 += 2;
            r1 += 2;
            r2 += 2;
            output_slice++;
 
          }

            r0 += tailstep;//1 + w;
            r1 += tailstep;//1 + w;
            r2 += tailstep;//1 + w;
      }
    }
  }


  return;
}



void max_pooling_forward_nchw_yolo_impl(struct CONV_ARG *conv_arg_pt, float *input, float *output)
{
  // offset
  const int input_hw = conv_arg_pt->input_h * conv_arg_pt->input_w;
  const int input_chw = conv_arg_pt->input_c * input_hw;
  const int output_hw = conv_arg_pt->output_h * conv_arg_pt->output_w;
  const int output_chw = conv_arg_pt->output_c * output_hw;

  int n, c, oh, ow, h, w;
  for (n = 0; n < conv_arg_pt->input_n; ++n) {
    for (c = 0; c < conv_arg_pt->input_c; ++c) {
      const float* input_slice = input + n * input_chw + c * input_hw;
      float* output_slice = output + n * output_chw + c * output_hw;
      for (oh = 0; oh < conv_arg_pt->output_h; ++oh) {
        for (ow = 0; ow < conv_arg_pt->output_w; ++ow) {
          int hs = oh * conv_arg_pt->str_h;
          int ws = ow * conv_arg_pt->str_w;
          int he = MIN(hs + conv_arg_pt->kernel_h, conv_arg_pt->input_h);
          int we = MIN(ws + conv_arg_pt->kernel_w, conv_arg_pt->input_w);
          int pool_index = oh * conv_arg_pt->output_w + ow;
          int max_index_slice = hs * conv_arg_pt->input_w + ws;
          for (h = hs; h < he; ++h) {
            for (w = ws; w < we; ++w) {
              int index = h * conv_arg_pt->input_w + w;
              if (input_slice[index] > input_slice[max_index_slice]) {
                max_index_slice = index;
              }
            }
          }
          output_slice[pool_index] = input_slice[max_index_slice];
        }
      }
    }
  }

  return;
}


void max_pooling_forward_nhwc_impl(struct  CONV_ARG * conv_arg_pt, float *input, float *output)
{

  int *max_index_slice=(int *)malloc(conv_arg_pt->input_c *sizeof(int));

  const int input_hwc = conv_arg_pt->input_h * conv_arg_pt->input_w * conv_arg_pt->input_c;
  const int output_hwc = conv_arg_pt->output_h * conv_arg_pt->output_w * conv_arg_pt->output_c;
  int n;
  for (n = 0; n < conv_arg_pt->input_n; ++n)
  {
    const float* input_slice = input + n * input_hwc;
    float* output_slice = output + n * output_hwc;
    int oh, ow, h, w, c;
    for (oh = 0; oh < conv_arg_pt->output_h; ++oh)
    {
      for (ow = 0; ow < conv_arg_pt->output_w; ++ow)
      {
        int hs = oh * conv_arg_pt->str_h - conv_arg_pt->pad_h;
        int ws = ow * conv_arg_pt->str_w - conv_arg_pt->pad_w;
        int he = MIN(hs + conv_arg_pt->kernel_h, conv_arg_pt->input_h);
        int we = MIN(ws + conv_arg_pt->kernel_w, conv_arg_pt->input_w);
        hs = MAX(hs, 0);
        ws = MAX(ws, 0);
        int pool_index = (oh * conv_arg_pt->input_w + ow) * conv_arg_pt->input_c;
        for (c = 0; c < conv_arg_pt->input_c; ++c)
        {
          max_index_slice[c] = (hs * conv_arg_pt->input_w + ws) * conv_arg_pt->input_c + c;
        }
        for (h = hs; h < he; ++h)
        {
          for (w = ws; w < we; ++w)
          {
            for (c = 0; c < conv_arg_pt->input_c; ++c)
            {
              int index = (h * conv_arg_pt->input_w + w) * conv_arg_pt->input_c + c;
              if (input_slice[index] > input_slice[max_index_slice[c]])
              {
                max_index_slice[c] = index;
              }
            }
          }
        }
        for (c = 0; c < conv_arg_pt->input_c; ++c)
        {
          output_slice[pool_index + c] = input_slice[max_index_slice[c]];
        }
      }
    }
  }

  free((void*)max_index_slice);

  return;
}


void ave_pooling_forward_nchw_impl(struct CONV_ARG * conv_arg_pt, float *input, float *output)
{
  const int input_hw = conv_arg_pt->input_h * conv_arg_pt->input_w;
  const int input_chw = conv_arg_pt->input_n * input_hw;
  const int output_hw = conv_arg_pt->output_h * conv_arg_pt->output_w;
  const int output_chw =conv_arg_pt->output_c * output_hw;
  int n, c, oh, ow, h, w;
  for (n = 0; n < conv_arg_pt->input_n; ++n)
  {
    for (c = 0; c < conv_arg_pt->input_c; ++c)
    {
      const float* input_slice = input + n * input_chw + c * input_hw;
      float* output_slice = output + n * output_chw + c * output_hw;
      for (oh = 0; oh < conv_arg_pt->output_h; ++oh)
      {
        for (ow = 0; ow < conv_arg_pt->output_w; ++ow)
        {
          int hs = MAX(oh * conv_arg_pt->str_h, 0);
          int ws = MAX(ow * conv_arg_pt->str_w, 0);
          int he = MIN(hs + conv_arg_pt->kernel_h, conv_arg_pt->input_h);
          int we = MIN(ws + conv_arg_pt->kernel_w, conv_arg_pt->input_w);
          int pool_index = oh * conv_arg_pt->output_w + ow;
          output_slice[pool_index]=0;
          for (h = hs; h < he; ++h)
          {
            for (w = ws; w < we; ++w)
            {
              int index = h * conv_arg_pt->input_w + w;
              output_slice[pool_index]+=input_slice[index];
            }
          }
          output_slice[pool_index] /= (he-hs)*(we-ws);
        }
      }
    }
  }

  return;
}



void ave_pooling_forward_nhwc_impl( struct CONV_ARG *conv_arg_pt, float *input, float *output )
{

  float *sum_slice=(float*)malloc(conv_arg_pt->input_c*sizeof(float));

  const int input_chw = conv_arg_pt->input_h * conv_arg_pt->input_w * conv_arg_pt->input_c;
  const int output_chw = conv_arg_pt->output_h * conv_arg_pt->output_w * conv_arg_pt->output_c;

  int n, oh, ow, c, h, w;
  for (n = 0; n < conv_arg_pt->input_n; ++n)
  {
    const float* input_slice = input + n * input_chw;
    float* output_slice = output + n * output_chw;
    for (oh = 0; oh < conv_arg_pt->output_h; ++oh)
    {
      for (ow = 0; ow < conv_arg_pt->output_w; ++ow)
      {
        const int hs = MAX(oh * conv_arg_pt->str_h, 0);
        const int ws = MAX(ow * conv_arg_pt->str_w, 0);
        const int he = MIN(hs + conv_arg_pt->kernel_h, conv_arg_pt->input_h);
        const int we = MIN(ws + conv_arg_pt->kernel_w, conv_arg_pt->input_w);
        const int pool_index = (oh * conv_arg_pt->output_w + ow) * conv_arg_pt->input_c;
        for (c = 0; c < conv_arg_pt->input_c; ++c)
        {
          sum_slice[c] = 0.0;
        }
        for (h = hs; h < he; ++h)
        {
          for (w = ws; w < we; ++w)
          {
            for (c = 0; c < conv_arg_pt->input_c; ++c)
            {
              int index = (h * conv_arg_pt->input_w + w) * conv_arg_pt->input_c + c;
              sum_slice[c] += input_slice[index];
            }
          }
        }
        for (c = 0; c < conv_arg_pt->input_c; ++c)
        {
          output_slice[pool_index + c] = sum_slice[c] / (he-hs)*(we-ws);
        }
      }
    }
  }
  free((void*)sum_slice);

  return;
}


void relu_forward_impl(float *input, float *output, int nchw, float slope)
{
  size_int i;
  if(slope==0.0)
  {
    for(i = 0; i < nchw; ++i)
    {
      output[i] = MAX(input[i],0);
    }
  }
  else
  {
    for(i = 0; i < nchw; ++i)
    {
      output[i] = MAX(input[i],0) + slope*MIN(input[i],0);
    }

  }

  return;
}



void power_forward_impl(float *input, float *output, int nchw, float power, float scale, float shift)
{
  float value;
  if(fabsf(scale)<1.0e-9)
  {
    if(fabs(power)<1.0e-9)
    {
      value=1.0;
    }
    else
    {
      value=powf(shift, power);
    }
    int i;
    for(i = 0; i < nchw; ++i)
    {
      output[i] = value;
    }
    return;
  }
  else
  {
    memcpy(output, input, sizeof(float)*nchw);
    int i;
    if(fabsf(scale-1.0)>1.0e-9)
    {
      for(i = 0; i < nchw; ++i)
      {
        output[i] *= scale;
      }
    }
    if(fabsf(shift)>1.0e-9)
    {
      for(i = 0; i < nchw; ++i)
      {
        output[i] += shift;
      }
    }
    if(fabsf(power-1.0)>1.0e-9)
    {
      for(i = 0; i < nchw; ++i)
      {
        output[i]=powf(output[i], power);
      }
    }
    return;
  }

  return;
}

void tanh_forward_impl(float *input, float *output, int nchw)
{
  int i;
  for(i = 0; i < nchw; ++i)
  {
    output[i] = tanh(input[i]);
  }

  return;
}


void sigmoid_forward_impl(float *input, float *output, int nchw)
{
  int i;
  for(i = 0; i < nchw; ++i)
  {
    output[i] = 1 / (exp(-input[i]) + 1);
  }

  return;
}


void softmax_forward_impl(float *input, float *output, int n, int chw)
{
  int i, j;
  for (i = 0; i < n; ++i)
  {
    float sum = 0;
    for (j = 0; j < chw; ++j)
    {
      int index = i * chw+ j;
      output[index] = exp(input[index]);
      sum += output[index];
    }
    for(j = 0; j < chw; ++j)
    {
      output[i * chw + j] /= sum;
    }
  }

  return;
}



void log_softmax_forward_impl(float *input, float *output, int n, int chw)
{
  int i, j;
  for(i=0; i<n; ++i)
  {
    float max=input[i*chw];
    for(j=0; j<chw; ++j)
    {
      int index = i*chw+j;
      if(input[index]>max)
        max=input[index];
    }
    float sum = 0;
    for(j=0; j<chw; ++j)
    {
      int index = i*chw+j;
      sum += exp(input[index]-max);
    }
    for(j=0; j<chw; ++j)
    {
      output[i*chw+j] = input[i*chw+j]-max-safe_log(sum);
    }
  }

  return;
}




void bias_forward_impl(float *input, float *bias, float *output, int k, int pq)
{
  int i, j;
  if(pq==1)
  {
    for(i = 0; i < k; ++i)
    {
      output[i] = input[i] + bias[i];
    }
    return;
  }

  for(i = 0; i < k; ++i)
  {
    for(j = 0; j < pq; ++j)
    {
      output[i*pq+j] = input[i*pq+j] + bias[i];
    }
  }

  return;
}



void batch_norm_forward_impl(float *input, float *output, float scale_factor, float *bn_scale1, float *bn_scale2, float eps, int n, int c, int h, int w)
{
  if(fabs(scale_factor)>1.0e-8)
  {
    scale_factor = 1.0/scale_factor;
  }
  else
  {
    scale_factor = 0;
  }

  float *mean, *var;
  mean = (float*)malloc(c*sizeof(float));
  var = (float*)malloc(c*sizeof(float));
  
  memset(mean, 0, sizeof(float)*c);
  memset(var, 0, sizeof(float)*c);

  int i, j, k;
  for(i=0; i<c; i++)
  {
	  if (bn_scale2[i] < 0)
		  bn_scale2[i] = 0;

    mean[i]=scale_factor*bn_scale1[i];
    var[i]=powf(scale_factor*bn_scale2[i]+eps, 0.5);
  }

  int hxw=h*w;
  for(i=0; i<n; i++)
  for(j=0; j<c; j++)
  for(k=0; k<hxw; k++)
  {
    output[i*c*hxw+j*hxw+k] = (input[i*c*hxw+j*hxw+k]-mean[j])/var[j];
  }

  free((void*)mean);
  free((void*)var);

  return;
}




void scale_forward_impl(float *input, float *output, float *gama, float *beta, int n, int c, int h, int w, bool bias)
{
  int hxw=h*w;
  int i, j, k;

  if(bias)
  {
    for(i=0; i<n; i++)
    for(j=0; j<c; j++)
    for(k=0; k<hxw; k++)
    {
      output[i*c*hxw+j*hxw+k] = input[i*c*hxw+j*hxw+k]*gama[j]+beta[j];
    }
  }
  else
  {
    for(i=0; i<n; i++)
    for(j=0; j<c; j++)
    for(k=0; k<hxw; k++)
    {
      output[i*c*hxw+j*hxw+k] = input[i*c*hxw+j*hxw+k]*gama[j];
    }
  }

  return;
}




void lrn_across_forward_impl(float *input, float *output, int local_size, float alpha, float beta, int channels, int height, int width)
{
  const int hxw = height*width;

  float *in_square=(float*)malloc(hxw*sizeof(float));
  memset(in_square, 0, hxw*sizeof(float));

  int i, j;
  for(i = 0; i < local_size/2; i++)
  {
    int idx = (height*i+0)*width+0;
    for(j=0; j<hxw; j++)
    {
      in_square[j]+=input[idx+j]*input[idx+j];
    }
  }

  int head = local_size/2;
  int tail = (int)(head)-(int)(local_size);
  const float alpha_div_size = alpha/local_size;

  for(i = 0; i < channels; i++, head++, tail++)
  {
    if(head < channels)
    {
      for(j=0; j<hxw; j++)
      {
        int idx = (height*head+0)*width+0;
        in_square[j] += input[idx+j]*input[idx+j];
      }
    }

    if(tail >= 0)
    {
      for(j=0; j<hxw; j++)
      {
        int idx = (height*tail+0)*width+0;
        in_square[j] -= input[idx+j]*input[idx+j];
      }
    }

    int idx = (height*i+0)*width+0;
    float *dst = output+idx;
    const float *src = input+idx;

    for(j = 0; j < hxw; j++)
    {
      dst[j] = src[j]*powf(1.0+alpha_div_size*in_square[j], -beta);
    }
  }

  free((void*)in_square);

  return;
}



void lrn_within_forward_impl(float *input, float *output, int local_size, float alpha, float beta, int channels, int height, int width)
{
  printf("not implemented\n");
  exit(1);

  return;
}



void elem_wise_operate_impl(int num_input, float **input, float *output, int len, enum OPERATION op)
{
  int i, j;

  switch(op)
  {
    case SUM:
      memset(output, 0, sizeof(float)*len);
      for(j=0; j<num_input; j++)
      {
        for(i=0; i<len; i++)
        {
          output[i] += input[j][i];
        }
      }
      break;
    case PROD:
      for(i=0; i<len; i++)
        output[i] = 1;
      for(j=0; j<num_input; j++)
      {
        for(i=0; i<len; i++)
        {
          output[i] *= input[j][i];
        }
      }
      break;
    case MAX:
      for(j=0; j<num_input; j++)
      {
        for(i=0; i<len; i++)
        {
          output[i] = MAX(input[j][i], output[i]);
        }
      }
      break;
    default: printf("unsupported operation\n");
  }

  free((void*)input);

  return;
}



void crop_forward_impl(float *input, float *output,  int axis, int n, int c, int h, int w, int on, int oc, int oh, int ow, int offset_n, int offset_c, int offset_h, int offset_w)
{
  int off_inp=0, off_oup=0;

  int i, j, k;
  for(i=0; i<on; i++)
  for(j=0; j<oc; j++)
  for(k=0; k<oh; k++)
  {
    int ii=i, jj=j, kk=k;
    switch(axis)
    {
      case 0: ii=i+offset_n;
      case 1: jj=j+offset_c;
      case 2: kk=k+offset_h; break;
      default: printf("unsupported axis\n");
    }

    off_inp=ii*c*h*w+jj*h*w+kk*w+offset_w;
    off_oup=i*oc*oh*ow+j*oh*ow+k*ow;
    memcpy(output+off_oup, input+off_inp, ow*sizeof(float));
  }

  return;
}




void unpack_no_dilation_nchw_impl( struct CONV_ARG * conv_arg_pt, const float* input, float* unpack)
{
  int unpack_index = 0;
  int c, r, s, p, q;
  for (c = 0; c < conv_arg_pt->input_c; ++c)
  {
    const int input_chw = c * conv_arg_pt->input_h * conv_arg_pt->input_w;
    const float* I_slice = input + input_chw;
    for (r = 0; r < conv_arg_pt->kernel_h; ++r)
    {
      for (s = 0; s < conv_arg_pt->kernel_w; ++s)
      {
        int r_offset = -conv_arg_pt->pad_h + r;
        for (p = 0; p < conv_arg_pt->output_h; ++p)
        {
          if (r_offset < 0 || r_offset >= (int)(conv_arg_pt->input_h))
          {
            for (q = 0; q < conv_arg_pt->output_w; ++q)
            {
              unpack[unpack_index++] = 0;
            }
          }
          else
          {
            int s_offset = -conv_arg_pt->pad_w + s;
            for (q = 0; q < conv_arg_pt->output_w; ++q)
            {
              if (s_offset < 0 || s_offset >= (int)(conv_arg_pt->input_w))
              {
                unpack[unpack_index++] = 0;
              }
              else
              {
                unpack[unpack_index++] = I_slice[r_offset * conv_arg_pt->input_w + s_offset];
              }
              s_offset += conv_arg_pt->str_w;
            }
          }
          r_offset += conv_arg_pt->str_h;
        }
      }
    }
  }
  return;
}




void unpack_dilation_nchw_impl( struct CONV_ARG * conv_arg_pt, const float* input, float* unpack)
{
  int unpack_index = 0;
  int c, r, s, p, q;
  for (c = 0; c < conv_arg_pt->input_c; ++c)
  {
    const int input_chw = c * conv_arg_pt->input_h * conv_arg_pt->input_w;
    const float* I_slice = input + input_chw;
    for (r = 0; r < conv_arg_pt->kernel_h; ++r)
    {
      for (s = 0; s < conv_arg_pt->kernel_w; ++s)
      {
        int R_offset = -conv_arg_pt->pad_h + r*conv_arg_pt->dila_h;
        for (p = 0; p < conv_arg_pt->output_h; ++p)
        {
          if (R_offset < 0 || R_offset >= (int)(conv_arg_pt->input_h))
          {
            for (q = 0; q < conv_arg_pt->output_w; ++q)
            {
              unpack[unpack_index++] = 0;
            }
          }
          else
          {
            int S_offset = -conv_arg_pt->pad_w + s*conv_arg_pt->dila_w;
            for (q = 0; q < conv_arg_pt->output_w; ++q)
            {
              if (S_offset < 0 || S_offset >= (int)(conv_arg_pt->input_w))
              {
                unpack[unpack_index++] = 0;
              }
              else
              {
                unpack[unpack_index++] = I_slice[R_offset * conv_arg_pt->input_w + S_offset];
              }
              S_offset += conv_arg_pt->str_w;
            }
          }
          R_offset += conv_arg_pt->str_h;
        }
      }
    }
  }
  return;
}



void unpack_no_dilation_nhwc_impl(struct CONV_ARG * conv_arg_pt, const float* input, float* unpack)
{
  int R_offset = -conv_arg_pt->pad_h;
  int p, q, h, w, c;
  for (p = 0; p < conv_arg_pt->output_h; ++p)
  {
    int S_offset = -conv_arg_pt->pad_w;
    for (q = 0; q < conv_arg_pt->output_w; ++q)
    {
      for (h = R_offset; h < (int)(conv_arg_pt->kernel_h) + R_offset; ++h)
      {
        for (w = S_offset; w < (int)(conv_arg_pt->kernel_w) + S_offset; ++w)
        {
          if (h >= 0 && h < (int)(conv_arg_pt->input_h) && w >= 0 && w < (int)(conv_arg_pt->input_w))
          {
            for(c = 0; c < conv_arg_pt->input_c; ++c)
            {
              unpack[c] = input[(h * conv_arg_pt->input_w + w) * conv_arg_pt->input_c + c];
            }
          }
          else
          {
            memset(unpack, 0, sizeof(float) * conv_arg_pt->input_c);
          }
          unpack += conv_arg_pt->input_c;
        }
      }
      S_offset += conv_arg_pt->str_w;
    }
    R_offset += conv_arg_pt->str_h;
  }
  return;
}



void pack_nchw_impl(struct CONV_ARG * conv_arg_pt, const float* unpack, float* input)
{
  int unpack_index = 0;
  int c, r, s, p, q;
  for (c = 0; c < conv_arg_pt->input_c; ++c)
  {
    const int input_chw = c * conv_arg_pt->input_h * conv_arg_pt->input_w;
    float* I_slice = input + input_chw;
    for (r = 0; r < conv_arg_pt->kernel_h; ++r)
    {
      for (s = 0; s < conv_arg_pt->kernel_w; ++s)
      {
        int R_offset = -conv_arg_pt->pad_h + r;
	for (p = 0; p < conv_arg_pt->output_h; ++p)
        {
          if (R_offset < 0 || R_offset >= (int)(conv_arg_pt->input_h))
          {
            unpack_index += conv_arg_pt->output_w;
          }
          else
          {
            int S_offset = -conv_arg_pt->pad_w + s;
            for (q = 0; q < conv_arg_pt->output_w; ++q)
            {
              if (S_offset >= 0 && S_offset < (int)(conv_arg_pt->input_w))
              {
                I_slice[R_offset * conv_arg_pt->input_w + S_offset] += unpack[unpack_index];
              }
              unpack_index++;
              S_offset += conv_arg_pt->str_w;
            }
          }
          R_offset += conv_arg_pt->str_h;
        }
      }
    }
  }
  return;
}

void pack_nhwc_impl(struct CONV_ARG * conv_arg_pt, const float* unpack, float* input)
{
  int p, q, h, w, c;
  int unpack_index = 0;
  int R_offset = -conv_arg_pt->pad_h;
  for (p = 0; p < conv_arg_pt->output_h; ++p)
  {
    int S_offset = -conv_arg_pt->pad_w;
    for (q = 0; q < conv_arg_pt->output_w; ++q)
    {
      for (h = R_offset; h < (int)(conv_arg_pt->kernel_h) + R_offset; ++h)
      {
        for (w = S_offset; w < (int)(conv_arg_pt->kernel_w) + S_offset; ++w)
        {
          if (h >= 0 && h < (int)(conv_arg_pt->input_h) && w >= 0 && w < (int)(conv_arg_pt->input_w))
          {
            float* I_slice = input + (h * conv_arg_pt->input_w + w) * conv_arg_pt->input_c;
            for(c = 0; c < conv_arg_pt->input_c; ++c)
            {
              I_slice[c] += unpack[unpack_index + c];
            }
          }
          unpack_index += conv_arg_pt->input_c;
        }
      }
      S_offset += conv_arg_pt->str_w;
    }
    R_offset += conv_arg_pt->str_h;
  }
  return;
}


void gemm( const float* A, const float* B, float* C, bool transa, bool transb, float alpha, float beta, int m, int n, int k)
{
#ifdef BLAS
  enum CBLAS_TRANSPOSE TransA = transa ? CblasTrans : CblasNoTrans;
  int lda = transa ? m : k;
  enum CBLAS_TRANSPOSE TransB = transb ? CblasTrans : CblasNoTrans;
  int ldb = transb ? k : n;
  cblas_sgemm(CblasRowMajor, TransA, TransB, m, n, k, alpha, A, lda, B, ldb, beta, C, n);
#else
  printf("not define the macro BLAS, but you are using the library, please define it in the Makefile\n");
  exit(1);
#endif
}


//m is the start position from zero, and n is the last position
void quick_sort_descend(float *arr, int start, int end)
{
  int i,j;
  float temp;    
  i=start;
  j=end;
  temp=arr[i];

  while(i<j)
  {
    while(i<j&&temp>arr[j])
      j--;
    if(i<j)
    {  
      arr[i]=arr[j];
      i++;
    }  
    while(i<j&&temp<arr[i])
      i++;
    if(i<j)
    {
      arr[j]=arr[i];  
      j--;  
    }
  }
  arr[i]=temp;
 
  if(start<i)
    quick_sort_descend(arr,start,i-1);
  if(i<end)
    quick_sort_descend(arr,i+1,end);

  return;
}




void tensor_concat_impl(int axis, int num_input, float **input, float *output, int *n, int *c, int *h, int *w, int no, int co, int ho, int wo)
{
  int i, j;
  if(axis==0)
  {
    float* p=output;
    for(i=0; i<num_input; i++)
    {
      int len=n[i]*c[i]*h[i]*w[i];
      memcpy(p, input[i], len*sizeof(float));
      p+=len;
    }
  }
  else if(axis==1)
  {
    int len3=co*ho*wo;
    for(i=0; i<no; i++)
    {
      float *pp=output+i*len3;
      for(j=0; j<num_input; j++)
      {
        int len1=c[j]*h[j]*w[j];
        memcpy(pp, input[j]+i*len1, len1*sizeof(float));
        pp += len1;
      }
    }
  }
  else if(axis==2)
  {
    printf("unsupported concact operation\n");
  }
  else if(axis==3)
  {
    printf("unsupported concact operation\n");
  }
  else
  {
    printf("unsupported concact operation\n");
  }

  return;
}


void reorg_forward_impl(float *x, int n, int c, int h, int w, int stride, int forward, float *out)
{
  int b,i,j,k;
  int out_c = c/(stride*stride);

  for(b = 0; b < n; ++b)
  {
    for(k = 0; k < c; ++k)
    {
      for(j = 0; j < h; ++j)
      {
        for(i = 0; i < w; ++i)
        {
          int in_index  = i + w*(j + h*(k + c*b));
          int c2 = k % out_c;
          int offset = k / out_c;
          int w2 = i*stride + offset % stride;
          int h2 = j*stride + offset / stride;
          int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));
          if(forward) out[out_index] = x[in_index];
          else out[in_index] = x[out_index];
        }
      }
    }
  }

  return;
}
