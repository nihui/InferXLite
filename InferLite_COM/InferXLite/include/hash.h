#ifndef HASH_H
#define HASH_H

#ifdef __cpluscplus
extern "C"
{
#endif //cpp

#include "inferxlite_common.h"

extern struct data_pipeline* data_map(const char *p, struct handler *hd);
extern struct data_pipeline* data_map_init(const char *p, int *nchw, struct handler *hd);
extern struct model_pipeline* weight_bias_map(const char *p, struct handler *hd);
extern struct model_pipeline* weight_bias_map_init(const char *p, int nw, int *nv, int *vw, struct data_arg dg, struct handler *hd);
extern uint64_t hash_str(const char* p, int n);
extern int hash_long_insert(const uint64_t n_hash, struct handler *hd);
extern int hash_long_get(const uint64_t n_hash, struct handler *hd);
extern void print_elem_value(int index, struct handler *hd);
extern void dltensor_type_init(DLTensor *tsr, int code, int bits, int lanes);

#ifdef __cplusplus
}
#endif
#endif
