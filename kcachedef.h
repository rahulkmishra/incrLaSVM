/*
 * kcachedef.h
 *
 *  Created on: 15-Apr-2013
 *      Author: rahul.mishra
 */

#ifndef KCACHEDEF_H_
#define KCACHEDEF_H_

#ifndef LASVM_KERNEL_T_DEFINED
#define LASVM_KERNEL_T_DEFINED
typedef double (*lasvm_kernel_t)(int i, int j, void* closure);
#endif

struct lasvm_kcache_s {
  lasvm_kernel_t func;
  void *closure;
  long long maxsize;
  long long cursize;
  int l;
  int *i2r;
  int *r2i;
  /* Rows */
  int    *rsize;
  float  *rdiag;
  float **rdata;
  int    *rnext;
  int    *rprev;
  int    *qnext;
  int    *qprev;
};



#endif /* KCACHEDEF_H_ */
