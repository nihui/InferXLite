
#include "inferxlite.h"

	int main()
	{
		char model_prefix_path[1000] = { "../data" };

		const int input_len = 1;
		const int input_w = 224;
		const int input_h = 224;

		const int loop_num = 1;
		float **loop_p;
		loop_p = (float**)malloc(sizeof(float*) * loop_num);

		// generate input data
		/*
		int i;
		for(i = 0; i < loop_p; i++)
		{
		  loop_p[i] = (float*) malloc( sizeof(float) *1*3*input_w*input_h );
		  memset(loop_p[i], 0, sizeof(float) *1*3*input_w*input_h);
		  int j;
		  for(j = 0; j < 1*3*input_len*input_len; j++)
			loop_p[i][j] = (rand()) / (float) RAND_MAX*255;
		}
		*/

		FILE *fp;
		errno_t err = fopen_s(&fp, "../data/im_row_col.dat", "r");
		//if ((fopen_s(&fp, "./data/deceleration_people_416.dat", "rt")) != false)
		if (err == 0)
			//if((fp=fopen("./data/416.dat", "rt"))!=NULL)
		{
			loop_p[0] = (float *)malloc(sizeof(float) * 1 * 3 * input_w * input_h);
			int i; int value;
			for (i = 0; i < 3 * input_w*input_h; i++)
			{
				
				fscanf_s(fp, "%f", loop_p[0] + i,sizeof(float));
				if (i < 100)
					;// printf(" %f\n", *(loop_p[0] + i));
			}


		}
		fclose(fp);
		

		inferx_context ctx1;
		inferx_init(model_prefix_path, &ctx1);

		// model example
		/*
		inferx_load("AlexNet", "an_1_3_227_227", &ctx1);
		inferx_load("AlexNet", "bn_1_3_227_227", &ctx2);
		inferx_load("YOLOv2", "cn_1_3_352_352", &ctx3);
		inferx_load("YOLOtiny", "cn_1_3_416_416", &ctx1);
		inferx_load("JDminiYOLOv2", "cn_1_3_416_416", &ctx1);
		inferx_load("YOLOtinyvoc", "cn_1_3_416_416", &ctx1);
		inferx_load("SqueezeNet", "dn_1_3_227_227", &ctx4);
		*/
		//inferx_load("JDminiYOLOv2", "cn_1_3_480_640", &ctx1);

		inferx_load("MOBILENETV2", "cn_1_3_224_224", &ctx1);
		float *pout = NULL;
		int i;
		for (i = 0; i < loop_num; i++)
		{
			/*printf(" %f\n", loop_p[i]);*/
			inferx_run(ctx1, loop_p[i], (void**)&pout);
		}

		//FILE *fpOut;//建立一个文件操作指针
		//fpOut = fopen("out.txt", "w");//


		//for (i = 0; i < 1000; i++) {
		//	    printf(fpOut, "%d   %f\n", i, pout[i]);
		//		fprintf(fpOut, "%d   %f\n", i, pout[i]);//同输出printf一样，以格式方式输出到文本中
		//}

		//fclose(fpOut);//关闭流
			
		//printf("print\n");
		//for(i=0; i<416 * 416 * 16; i++)
		//	printf("%d   %f\n", i, pout[i]);

		inferx_clear(ctx1);
		system("pause");
		return 0;
	}

