#include "common.clh"

__kernel void rc1(__global SCALAR *rc, 
				  __global SCALAR *piT, 
				  __global SCALAR *A, 
				  int m,
				  int n) 
{
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
	if (i < n) {
		// rc = 0 - piT A
		SCALAR piTA = 0.0;
		for (int k = 0; k < m; k++) {
			piTA += piT[k] * A[i + k*n];
		}
		
		rc[i] = 0.0 - piTA;
	}
}