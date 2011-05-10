#include "C:/Users/Iain/LPG/libLPG/kernel/common.clh"

__kernel void rc2(__global SCALAR *rc, 
				  __global SCALAR *c, 
				  __global SCALAR *piT, 
				  __global SCALAR *A, 
				  int m,
				  int n) 
{
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
	if (i < n) {
		// rc = c - piT A
		SCALAR piTA = 0.0;
		for (int k = 0; k < m; k++) {
			//piTA += piT[k] * A[i + k*n];
			piTA += piT[k] * A[k + i*m];
		}
		
		rc[i] = c[i] - piTA;
	}
}