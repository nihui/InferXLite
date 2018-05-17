#include "inferxlite_common.h"
#include "interface.h"
#include "model_init.h"
#include "string.h"
#include "pipe.h"



void inferx_init(char *path, inferx_context *ctx)
{
  handler_init(ctx);

  func_pointer_init(path, &(ctx->hd));

  pipeline_init(&(ctx->hd));

  model_init(&(ctx->hd));

  return;
}


void inferx_run(inferx_context ctx, void *p, void **pout)
{
  if(ctx.tag!=123456)
  {
    printf("the ctx is not initialized, you can not use it\n");
  }
  ctx.hd.tsw=time_forward;
  
  ctx.func(ctx.hd.fpm.path, ctx.model, ctx.data, p, pout, &(ctx.hd));

  if(ctx.hd.tsw==time_layer)
    print_time_by_layer(&(ctx.hd));


  return;
}


void inferx_load(char *model, char *data, inferx_context *ctx)
{
  char fname[1000];
  sprintf(fname, "%s/%s", ctx->hd.fpm.path, model);

  printf("loading model....\n");

  if(!inferx_var_add_init(model,&(ctx->hd)))
  {
    if(model)
    {
      //load_model_and_data_from_ball(fname, &(ctx->hd));
      load_model_from_binary(fname, &(ctx->hd));
    }
  }

  int i;
  for(i=0; i<ctx->hd.fpm.len; i++)
  {
    if(!strcmp(ctx->hd.fpm.name[i], model)) 
    {
      ctx->func = ctx->hd.fpm.func[i];
      ctx->model = (char *)malloc(sizeof(char)*1000);
      strcpy(ctx->model, model);
      ctx->data = (char *)malloc(sizeof(char)*1000);
      strcpy(ctx->data, data);
      break;
    }
  }

  ctx->func(ctx->hd.fpm.path, ctx->model, ctx->data, NULL, NULL, &(ctx->hd));
  
  ctx->tag = 123456;

  return;
}


void inferx_clear(inferx_context ctx)
{
  if(ctx.tag != 123456)
  {
    printf("you are clearing a non-existing context\n");
    exit(1);
  }
  free((void*)ctx.model);
  free((void*)ctx.data);

  return;
}
