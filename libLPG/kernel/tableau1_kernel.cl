#include "C:/Users/Iain/Dev/LPG/libLPG/kernel/common.clh"

__kernel void tableau1(	__global double *Binv,
						__global double *BinvAs,
						int m, int r) 
{
	// Get the index of the current element to be processed
	int i = get_global_id(0);
	
	if (i < m) {
		double erBinvAs = BinvAs[r];
			
		if (i != r) {
			double eiBinvAs = BinvAs[i];
			for (int j = 0; j < m; j++) {
				Binv[j+i*m] = Binv[j+i*m] - (eiBinvAs)/(erBinvAs)*Binv[j+r*m];
			}
		}
	}
}