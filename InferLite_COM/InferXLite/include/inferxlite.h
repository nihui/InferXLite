#pragma once

#ifdef __cplusplus
extern "C"
{
#endif //cpp

#include "inferxlite_common.h"

extern void* inferx_get_data(char *bottom_pre, char *data_pre, struct handler *hd);
//extern void init_func_pointer();
/**
 * @brief  This function is for initialiization.
 * @detail Initialize the network,build context to index network, build mapping from layername to address
 * @param  path The path which config file and data file lie in
 * @param  ctx The context to index network 
*/
extern void inferx_init(char *path, inferx_context *ctx);
/** @brief  This function load data for running network
 *  @detail Load the weight data and allocate memory for feature map 
 *  @param  model The data file name is loaded.
 *  @param  data The string contain intput data tag and input data shape
 *  @param  ctx  The context to index network 
*/
extern void inferx_load(char *model, char *data, inferx_context *ctx);
/** @brief  Run the network
 *  @detail Input the data, run the network
 *  @param  ctx  The context to index network
 *  @param  p  The input data pointer
 *  @param  pout The pointer point to output data pointer
*/
extern void inferx_run(inferx_context ctx, void *p, void **pout);
/** @brief  Clear the network
 *  @datail Clear the memory of network
 *  @param  ctx  The context index network
*/
extern void inferx_clear(inferx_context ctx);

#ifdef __cplusplus
}
#endif //cpp

