#include "C:/Users/Iain/Dev/LPG/libLPG/kernel/common.clh"

__kernel void dual(	__global double *cBT, 
					__global double *Binv, 
					__global double *piT, 
					int m) 
{
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
	if (i < m) {
	 
		// piT = cbT Binv
		double value = 0.0;
		for (int k = 0; k < m; k++) {
			value = value + cBT[k] * Binv[i+k*m];
		}
		
		piT[i] = value;
		
	}
}