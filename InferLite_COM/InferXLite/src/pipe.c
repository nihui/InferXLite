#include "pipe.h"
#include "metafmt.h"
#include <math.h>
#include <dirent.h>
#include <stdlib.h>
#include <stdio.h>

void func_pointer_init(char *path, struct handler *hd)
{
  int len =strlen(path);
  hd->fpm.path = (char*)malloc(len);
  strcpy(hd->fpm.path, path);

  hd->is_has_init=(char **)malloc(sizeof(char*)*hd->len_init);
  memset(hd->is_has_init, 0, sizeof(char*)*hd->len_init);


  hd->fpm.name = (char **)malloc(sizeof(char*)*hd->max_num_func);
  hd->fpm.len = 0;
  int i;
  for(i=0; i<hd->max_num_func; i++)
  {
    hd->fpm.name[i]=(char*)malloc(sizeof(char)*100);
  }

  hd->fpm.func = (model_func_pointer *)malloc(sizeof(model_func_pointer)*hd->max_num_func);

  memset(hd->fpm.func, 0, sizeof(model_func_pointer)*hd->max_num_func);

  return;
}




void handler_init(inferx_context *ctx)
{
  ctx->hd.max_num_pipeline=1000;
  ctx->hd.is_update_input=false;
  ctx->hd.tsw=time_no;
  ctx->hd.time_head=NULL;
  ctx->hd.time_tail=NULL;
  ctx->hd.time_layer_cnt=0;
  ctx->hd.max_num_func=100;
  ctx->hd.len_init=100;
  ctx->hd.len_elem=0;
  ctx->hd.elem=NULL;
  
#ifndef PERFDNN_CL
  ctx->hd.dvct=kCPU;
#elif defined(PERFDNN_CL)
  ctx->hd.dvct=kOpenCL;
#endif
  
  ctx->hd.cw=conv_nature;
  //ctx->hd.cw=conv_gemm;
  //ctx->hd.cw=conv_perfdnn;

  return;
}



void pipeline_init(struct handler *hd)
{
  hd->weight_has_load=false;
  hd->data_has_init=false;
  srand(1);

  hd->time_tail=(struct TIMEBYLAYER *)malloc(sizeof(struct TIMEBYLAYER));
  hd->time_tail->tp=NULL;


  hd->modelflow=(struct model_pipeline*)malloc(sizeof(struct model_pipeline)*hd->max_num_pipeline);
  hd->dataflow=(struct data_pipeline*)malloc(sizeof(struct data_pipeline)*hd->max_num_pipeline);

#ifdef CUDNN
  cudnnStatus_t status;
  status = cudnnCreate(&(hd->hCudNN));
  if (status != CUDNN_STATUS_SUCCESS)
    printf("cudnn error %d\n", status);
#endif
#ifdef PERFDNN_OCL
    hd->perf_ocl_context = perf_dnn_setup_cl(PERFDNN_OCL_AMD, PERFDNN_OCL_GPU, 0);
#endif

  int i;
  for(i=0; i<hd->max_num_pipeline; i++)
  {
    hd->modelflow[i].weight=NULL;
  }

  return;
}


void weight_bias_rand_init(struct model_pipeline* m, int nw, struct handler *hd)
{
  int i;
  for(i=0; i<nw; i++)
  {
    meta_rand_init(m->weight+i, hd);
  }

  return;
}


void input_rand_init(struct data_pipeline* input, struct handler *hd)
{
  meta_rand_init(&(input->datas), hd);
   
  return;
}


struct model_pipeline* get_model(char *name, int nw, int *nv, int *vw, struct data_arg dg, struct handler *hd)
{
  struct model_pipeline* model;
  if(hd->weight_has_load)
  {
    model=weight_bias_map(name, hd);
    check_size(model, nw, nv, vw, name);
  }
  else
  {
    model=weight_bias_map_init(name, nw, nv, vw, dg, hd);
    weight_bias_rand_init(model, nw, hd);
  }

  return model;
}


void check_size(struct model_pipeline* model, int nw, int *nv, int *vw, char *name)
{
  enum checkVar{strict=0, efficient=1};

  enum checkVar way=efficient;

  if(way==strict)
  {
    if(nw==model->nw)
    {
      int i, vw_i=0;
      for(i=0; i<nw; i++)
      {
        if(nv[i]!=model->weight[i].ndim)
        {
          printf("the model input arguments has some mistake for nv %d vs %d for the layer %s, check it\n", nv[i], model->weight[i].ndim, name);
          exit(1);
        }
        else
        {
          int j;
          for(j=0; j<nv[i]; j++)
          {
            if(model->weight[i].shape[j]!=vw[vw_i++])
            {
              printf("the model input arguments has some mistake for nw %d nv %d vw %d vs %d for the layer %s, check it\n", i, j, vw[vw_i-1], model->weight[i].shape[j], name);
              exit(1);
            }
          }
        }
      }
    }
    else
    {
      printf("the model input arguments has some mistake for nw %d vs %d for the layer %s, check it\n", nw, model->nw, name);
      exit(1);
    }
  } 
  else
  {
    if(nw==model->nw)
    {
      int i, j, vw_i=0;
      for(i=0; i<nw; i++)
      {
        int len=1;
        for(j=0; j<nv[i]; j++)
          len*=vw[vw_i++];
        int tsr_len=1;
        for(j=0; j<model->weight[i].ndim; j++)
          tsr_len*=model->weight[i].shape[j];
        if(len!=tsr_len)
        {
          printf("the model input arguments has some mistake for efficient length %d vs %d for the layer %s, check it\n", len, tsr_len, name);
          exit(1);
        }
      }
    }
    else
    {
      printf("the model input arguments has some mistake for nw %d vs %d for the layer %s, check it\n", nw, model->nw, name);
      exit(1);
    }
  }

  return;
}


void load_model_and_data_from_ball(char *fname_pre, struct handler *hd)
{
  char fname[1000];
  sprintf(fname, "%s.dat", fname_pre);

  FILE *fpb;
  if((fpb=fopen(fname, "rt"))==NULL)
  {
    printf("can not open the file %s\n", fname);
    exit(1);
  }

  char prefix_file[1000];
  memset(prefix_file, 0, sizeof(char)*1000);

  int len=strlen(fname_pre);
  char *p=fname_pre+len;
  int j;
  for(j=0; j<len; j++)
  {
    if(*p == '/')
    {
      p++;
      break;
    }
    p--;
  }

  
  if(strcmp(p, "ball"))
  {
    strcpy(prefix_file, p);
  }
  else
  {
    printf("to avoid the namespace mistake, please name the ball file to model name\n");
  }


  int cnt_obj;
  fscanf(fpb, "%100d\n", &cnt_obj);
  int l;
  for(l=0; l<cnt_obj; l++)
  {
    char keyname[1000];
    fscanf(fpb, "%s", keyname);
    if(strlen(keyname)>3&&keyname[0]=='d'&&keyname[1]=='a'&&keyname[2]=='t'&&keyname[1]=='a')
    {
      int i;
      int nchw[4], len;
      for(i=0; i<4; i++)
      {
        fscanf(fpb, "%d", nchw+i);
      }
      len=nchw[0]*nchw[1]*nchw[2]*nchw[3];
      //printf("data %d\n", len);
      struct data_pipeline* in=data_map_init(keyname, nchw, hd);
      float *tmp=(float *)malloc(sizeof(float)*len);
      for(i=0; i<len; i++)
      {
        fscanf(fpb, "%f", tmp+i);
      }
      meta_float_to_tensor(tmp, in->datas, hd);
      free((void*)tmp);
    }
    else
    {
      int nw, *nv, *vw;
      int i, j, k;
      fscanf(fpb, "%d", &nw);
      nv = (int*)malloc(sizeof(int)*nw); 
      for(i=0; i<nw; i++)
      {
        fscanf(fpb, "%d", nv+i);
      }
      int sum=0;
      for(k=0; k<nw; k++)
      {
        sum+=nv[k];
      }
      vw = (int*)malloc(sizeof(int)*sum);
      for(i=0; i<sum; i++)
      {
        fscanf(fpb, "%d", vw+i);
      }
      struct data_arg dg;
      dg.uo=-1;

      char real_name[1000];
      sprintf(real_name, "%s%s", prefix_file, keyname);
      struct model_pipeline* model=weight_bias_map_init(real_name, nw, nv, vw, dg, hd);

      for(i=0; i<nw; i++)
      {
        int len=1;
        for(j=0; j<model->weight[i].ndim; j++)
	{
          len*=model->weight[i].shape[j];
        }
        float *tmp=(float *)malloc(sizeof(float)*len);
        for(j=0; j<len; j++)
        {
          fscanf(fpb, "%f", tmp+j);
        }
        meta_float_to_tensor(tmp, model->weight[i], hd);
        free((void*)tmp);
      }
      free((void*)nv);
      free((void*)vw);
    }
  }

  fclose(fpb);

  return;
}


void load_model_from_binary(char *fname_pre, struct handler *hd)
{
  char fname[1000];
  sprintf(fname,"%s.dat",fname_pre);
  
  FILE *fpb;
  if((fpb=fopen(fname,"rb"))==NULL)
  {
    printf("can not open the file %s\n", fname);
    exit(1);
  }
  
  char prefix_file[1000];
  memset(prefix_file, 0, sizeof(char)*1000);
  
  
  
  int len=strlen(fname_pre);
  char *p=fname_pre+len;
  int j;
  for(j=0; j<len; j++)
  {
    if(*p == '/')
    {
      p++;
      break;
    }
    p--;
  }
  
  if(strcmp(p, "ball"))
  {
    strcpy(prefix_file, p);
  }
  else
  {
    printf("to avoid the namespace mistake, please name the ball file to model name\n");
    exit(1);
  }
  
  
  
  int cnt_obj;
  fread(&cnt_obj, sizeof(int),1,fpb);
  //printf("cnt_obj %d\n",cnt_obj);
  for(int l=0; l<cnt_obj; l++)
  {
    int keyname_length;
    //get keyname length
    fread(&keyname_length, sizeof(int),1,fpb);
	
    //printf("keyname_length %d\n",keyname_length);
	char *keyname = (char *)malloc(sizeof(char)* keyname_length);
    //char keyname[keyname_length];
    //int num_char;
    //fread(&num_char,sizeof(int),1,fpb);
    for(int ss=0; ss<keyname_length; ++ss)
    { 
      int number_char,read_num;
      number_char=0;
      read_num=fread(&number_char,sizeof(int),1,fpb);
      if(number_char==0)
      {
        fseek(fpb,-1,SEEK_CUR);
        int tmp_test;
        read_num=fread(&tmp_test,sizeof(int),1,fpb);
        printf("tmp_test %d\n",tmp_test);
        if(ss!=0)
        {
          ss-=1;
        }
        continue;
      }
    	
      //printf("readNum: %d\n",readNum);
      //printf("cnt_obj num: %d\n",l);
      //printf("char num:%d\n",number_char);
      keyname[ss]=(char) number_char;
      //printf("keyname %d: %c\n",ss,keyname[ss]);
    }
    keyname[keyname_length]='\0';
    printf("keyname %s\n",keyname);
    int nw_int, *nv_int, *vw_int;
    int i, j, k;
    fread(&nw_int,sizeof(int),1,fpb);
    //printf("nw_int %d\n",nw_int);
    nv_int = (int*)malloc(sizeof(int)*nw_int);
    for(i=0; i<nw_int; ++i)
    {
      fread(nv_int+i,sizeof(int),1,fpb);
      //printf("nv_int[%d]: %d\n",i,*(nv_int+i));
    }
    int sum=0;
    for(k=0; k<nw_int; ++k)
    {
      sum+=nv_int[k];
    }
    //printf("sum: %d\n",sum);
    vw_int =(int*)malloc(sizeof(int)*sum);
    for(i=0;i<sum; ++i)
    {
      //printf("test\n");
      fread(vw_int+i,sizeof(int),1,fpb);
      //printf("vw_int[%d]: %d\n",i,*(vw_int+i));
    }
    int nw, *nv, *vw;
    nw = (int) nw_int;
    //printf("nw %d\n",nw);
    nv = (int*)malloc(sizeof(int)*nw);
    for(int i =0; i<nw; ++i)
    {
      nv[i]=(int) nv_int[i];
      //printf("nv[%d]: %d\n",i,nv[i]);
    }
    vw = (int*)malloc(sizeof(int)*sum);
    for(int i=0; i<sum; ++i)
    {
      vw[i]=(int) vw_int[i];
      //printf("vw[%d]: %d\n",i,vw[i]);
    }
    
    struct data_arg dg;
    dg.uo=-1;
    char real_name[1000];
    sprintf(real_name, "%s%s", prefix_file, keyname);
    struct model_pipeline* model=weight_bias_map_init(real_name,nw,nv,vw,dg,hd);
    
    for(i=0; i<nw; i++)
    {
      int len=1;
      for(j=0;j<model->weight[i].ndim;++j)
      {
        len*=model->weight[i].shape[j];
        printf("shape[%d]: %d\n",j,model->weight[i].shape[j]);
      }
      //printf("len = %d: \n",len);
      float *tmp = (float *)malloc(sizeof(float)*len);
      float tmp_weight;
      for(j=0; j<len; j++)
      {
        fread(&tmp_weight,sizeof(float),1,fpb);
        //printf("%f\n",tmp_weight);
        tmp[j]=(float) tmp_weight;
        //printf("%f\n",tmp_weight);
        //printf("%f\n",tmp[j]);
        //sprintf(tmp_weight,"%f",tmp[j]);
      }
      //printf("%d\n",l);
      meta_float_to_tensor(tmp,model->weight[i], hd);
      
		  //for (int m = 0; m<len; ++m)
		  //{
			 // if (m < 100) {
				//  printf("-----[%d] %f\n", m, tmp[m]);
				//  fflush(stdout);

			 // }

		  //}

      
      free((void*)tmp);
    }
    free((void*)nv);
    free((void*)vw);
    free((void*)nv_int);
    free((void*)vw_int);
  }
  fclose(fpb);

  return;	
}



void print_time_by_layer(struct handler *hd)
{
  struct TIMEBYLAYER *p=hd->time_head;
  if(!p)
    return;
  while(p->tp)
  {
    p->time/=(double)(hd->time_layer_cnt);
    printf("%s %f\n", p->tname, p->time);
    p=p->tp;
  }

  return;
}



int inferx_get_data_len(DLTensor input, int *nchw, int output_c, int pad_h, int pad_w, int kernel_h, int kernel_w, int str_h, int str_w, char func)
{
  int h1, w1;
  if(func==0)//it means convolution
  {
    h1=(input.shape[2]+2*pad_h-kernel_h)/str_h+1;
    w1=(input.shape[3]+2*pad_w-kernel_w)/str_w+1;
  }
  else//it means pooling
  {
    h1=ceilf((float)(input.shape[2]+2*pad_h-kernel_h)/(float)str_h)+1;
    w1=ceilf((float)(input.shape[3]+2*pad_w-kernel_w)/(float)str_w)+1;
  }

  nchw[0]=input.shape[0];
  nchw[1]=output_c;
  nchw[2]=h1;
  nchw[3]=w1;

  return h1*w1*input.shape[0]*output_c;
}



void inferx_update_data_shape(struct data_pipeline *input, struct data_pipeline *output, int output_c, int pad_h, int pad_w, int kernel_h, int kernel_w, int str_h, int str_w, int dila_h, int dila_w, char func)
{
  if(func==0)//it means convolution
  {
    output->datas.shape[2]=(input->datas.shape[2]+2*pad_h-(dila_h*(kernel_h-1)+1))/str_h+1;
    output->datas.shape[3]=(input->datas.shape[3]+2*pad_w-(dila_w*(kernel_w-1)+1))/str_w+1;
  }
  else//it means pooling
  {
    output->datas.shape[2]=ceilf((float)(input->datas.shape[2]+2*pad_h-kernel_h)/str_h)+1;
    output->datas.shape[3]=ceilf((float)(input->datas.shape[3]+2*pad_w-kernel_w)/str_w)+1;
  }
  output->datas.shape[0]=input->datas.shape[0];
  output->datas.shape[1]=output_c;

  return;
}

void inferx_keep_data_shape(struct data_pipeline *input, struct data_pipeline *output)
{
  int i;

  for(i=0; i<4; i++)
  {
    output->datas.shape[i]=input->datas.shape[i];
  }

  return;
}


void inferx_zero_data(struct data_pipeline *p)
{
  int len=p->datas.shape[0]*p->datas.shape[1]*p->datas.shape[2]*p->datas.shape[3]*p->datas.dtype.bits*p->datas.dtype.lanes/8;
  memset(p->datas.data, 0, len);

  return;
}


void inferx_parse_str(char *data, int *nchw)
{
  char data_c[1000];
  char *p=data;
  int index=0;
  while(*p != 0)
  {
    if ((*p >= '0') && (*p <= '9'))
    {
      break;
    }
    else
    {
      data_c[index++]=*p;
      p++;
    }
  }
  data_c[index]='\0';
  
  inferx_str_to_int(p, nchw);

  strcpy(data, data_c);
 
  return;
}


void inferx_str_to_int(const char *nchw_c, int *nchw_l)
{
  *nchw_l=0;
  while(*nchw_c!= 0)
  {
    if ((*nchw_c>= '0') && (*nchw_c<= '9'))
    {
      *nchw_l = *nchw_l * 10 + (*nchw_c- '0');
      nchw_c++;
    }
    else
    {
      inferx_str_to_int(++nchw_c, ++nchw_l);
      break;
    }
  }

  return;
}


void inferx_update_input_data(char *bottom, struct handler *hd)
{
  struct data_pipeline* input=data_map(bottom, hd);
  int i;
  if(input->datas.dtype.code==2&&input->datas.dtype.lanes==1)
  {
    float *p=(float*)input->datas.data;
    int len=input->datas.shape[0]*input->datas.shape[1]*input->datas.shape[2]*input->datas.shape[3];
    for(i=1; i<len; i++)
    {
      p[i]+=p[i-1]+p[len-i];
      p[i] /= 3.0;
    }
  }

  return;
}


void inferx_update_layer_timer(struct timeval end_layer, char *mname, struct handler *hd)
{
  if(!hd->time_tail->tp||hd->time_tail->tp->tp==hd->time_tail)
  {
    struct TIMEBYLAYER *p=(struct TIMEBYLAYER *)malloc(sizeof(struct TIMEBYLAYER));
    if(hd->time_head)
    {
      hd->time_tail->tp->tp=p;
    }
    else
    {
      hd->time_head=p;
    }
    p->tp=hd->time_tail;
    hd->time_tail->tp=p;
    p->tname=(char*)malloc(sizeof(char)*strlen(mname));
    strcpy(p->tname, mname);
    p->time=(end_layer.tv_sec-hd->start_layer.tv_sec)*1000 + (double)(end_layer.tv_usec-hd->start_layer.tv_usec)/1000;
  }
  else
  {
    hd->time_tail->time+=(end_layer.tv_sec-hd->start_layer.tv_sec)*1000 + (double)(end_layer.tv_usec-hd->start_layer.tv_usec)/1000;
    hd->time_tail=hd->time_tail->tp;
  }

  return;
}

void inferx_insert_model_func(char *name, model_func_pointer func, struct handler *hd)
{
  if(hd->fpm.len>=hd->max_num_func)
  {
    printf("the model func pointer vector space is not enough %d %d, please enlarge it\n", hd->fpm.len, hd->max_num_func);
    exit(1);
  }
  strcpy(hd->fpm.name[hd->fpm.len], name);
  hd->fpm.func[hd->fpm.len]=func;

  hd->fpm.len++;
   
  return;
}


void inferx_set_init_var(bool *weight_has_load, bool *data_has_init, char *weight, char *data, struct handler *hd)
{
  int i;

  for(i=0; i<hd->len_init; i++)
  {
    if(hd->is_has_init[i]==NULL)
      break;
    if(weight && !strcmp(hd->is_has_init[i], weight))
    {
      *weight_has_load=true;
    }
  }

  return;
}



bool inferx_var_add_init(char *var, struct handler *hd)
{
  int i;
  for(i=0; i<hd->len_init; i++)
  {
    if(hd->is_has_init[i])
    {
      if(!strcmp(hd->is_has_init[i], var))
      {
        return true;
      }
    }
    else
    {
      hd->is_has_init[i]=(char *)malloc(sizeof(char)*100);
      strcpy(hd->is_has_init[i], var);
      return false;
    }
  }
  
  if(i==hd->len_init)
  {
    printf("the lenInit is too small, please enlarge it\n");
  }

  return false;
}
