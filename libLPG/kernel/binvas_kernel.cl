#include "common.clh"

__kernel void binvas(	__global SCALAR *BinvAs, 
						__global SCALAR *Binv,
						__global SCALAR *A, 
						int m, int n, int s) 
{
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
    if (i < m) {
 
		// BinvAs = Binv * As
		SCALAR value = 0.0;
		for (int k = 0; k < m; k++) {
			value += Binv[k + i*m] * A[s+k*n];
		}
		
		BinvAs[i] = value;
		
	}
}