#ifndef META_META_H_
#define META_META_H_

#include "inferxlite_common.h" 
#ifdef __cplusplus
extern "C"
{
#endif //cpp

extern void meta_rand_init(DLTensor *p, struct handler *hd);
extern void meta_float_to_tensor(float *input, DLTensor tnsr, struct handler *hd);

#ifdef __cpluplus
}
#endif //cpp
#endif  // META_META_H
