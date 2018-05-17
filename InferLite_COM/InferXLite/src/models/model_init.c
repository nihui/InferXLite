#include "model_init.h"
#include "pipe.h"
#include "YOLOtiny.h"
#include "JDminiYOLOv2.h"
#include "SqueezeNet.h"
#include "ShipSSD.h"
#include "MOBILENETV2.h"

void model_init(struct handler *hd)
{
	//inferx_insert_model_func("YOLOtiny",YOLOtiny, hd);
	//inferx_insert_model_func("JDminiYOLOv2",JDminiYOLOv2, hd);
	//inferx_insert_model_func("SqueezeNet", SqueezeNet, hd);
	//inferx_insert_model_func("ShipSSD", ShipSSD, hd);
	inferx_insert_model_func("MOBILENETV2", MOBILENETV2, hd);
	return;
}
