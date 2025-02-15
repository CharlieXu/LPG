#include "C:/Users/Iain/Dev/LPG/libLPG/kernel/common.clh"

__kernel void binvas(	__global double *BinvAs, 
						__global double *Binv,
						__global double *A, 
						int m, int n, int s) 
{
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
    if (i < m) {
 
		// BinvAs = Binv * As
		double value = 0.0;
		for (int k = 0; k < m; k++) {
			//value += Binv[k + i*m] * A[s+k*n];
			value += Binv[k + i*m] * A[k+s*m];
		}
		
		BinvAs[i] = value;
		
	}
}