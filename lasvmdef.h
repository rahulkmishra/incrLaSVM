/*
 * structdef.h
 *
 *  Created on: 12-Apr-2013
 *      Author: rahul.mishra
 */

#ifndef STRUCTDEF_H_
#define STRUCTDEF_H_


#if USE_FLOAT
# define real_t float
#else
# define real_t double
#endif


struct lasvm_s
{
	lasvm_kcache_t *kernel;
	int     sumflag;
	real_t  cp;
	real_t  cn;
	int     maxl;
	int     s;
	int     l;
	real_t *alpha;
	real_t *cmin;
	real_t *cmax;
	real_t *g;
	real_t  gmin, gmax;
	int     imin, imax;
	int     minmaxflag;
};





#endif /* STRUCTDEF_H_ */
